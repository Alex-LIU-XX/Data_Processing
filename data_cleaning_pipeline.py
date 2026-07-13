import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import median_filter
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional


# ─── 数据结构定义 ───────────────────────────────────────────────────────────

@dataclass
class CleaningConfig:
    """各阶段的阈值配置（支持按数据集/本体类型覆盖）"""
    # S1: 突变检测
    s1_residual_std_mult: float = 3.0
    s1_accel_std_mult: float = 3.0
    s1_jerk_std_mult: float = 3.0
    s1_medfilt_kernel: int = 5
    s1_sg_window: int = 11
    s1_sg_order: int = 3

    # S2: 趋势对齐
    s2_da_threshold: float = 0.6
    s2_sg_window: int = 11
    s2_sg_order: int = 3
    s2_max_lag: int = 10

    # S3: 极值过滤
    s3_alpha: float = 1.5
    s3_gripper_exempt: bool = True

    # S4: FK一致性（暂无URDF时使用统计校验）
    s4_pos_tolerance: float = 0.1
    s4_rot_tolerance: float = 0.2

    # 综合
    episode_discard_on_corrupt: bool = False


@dataclass
class CleaningReport:
    """各阶段的清洗统计"""
    stage_logs: dict = field(default_factory=dict)

    def add(self, stage: str, msg: str, detail: Optional[dict] = None):
        entry = {"msg": msg, "detail": detail or {}}
        if stage not in self.stage_logs:
            self.stage_logs[stage] = []
        self.stage_logs[stage].append(entry)

    def summary(self) -> str:
        lines = []
        for stage, entries in self.stage_logs.items():
            dropped = sum(
                e["detail"].get("dropped_frames", 0) + e["detail"].get("dropped_episodes", 0) * 200
                for e in entries if "detail" in e
            )
            lines.append(f"  {stage}: {len(entries)} ops, ~{dropped} frames affected")
        return "\n".join(lines)


# ─── 数据加载 ───────────────────────────────────────────────────────────────

class LeRobotDataset:
    """读取 LeRobot v3.0 格式数据集"""

    def __init__(self, root: str):
        self.root = Path(root)
        self.meta_dir = self.root / "meta"
        self.data_dir = self.root / "data"

        with open(self.meta_dir / "info.json") as f:
            self.info = json.load(f)
        with open(self.meta_dir / "action_patch_info.json") as f:
            self.action_info = json.load(f)
        with open(self.meta_dir / "stats.json") as f:
            self.stats = json.load(f)

        self.robot_type = self.info["robot_type"]
        self.fps = self.info["fps"]
        self.total_episodes = self.info["total_episodes"]
        self.total_frames = self.info["total_frames"]

        self.df: Optional[pd.DataFrame] = None
        self.episodes_df: Optional[pd.DataFrame] = None

    def load(self):
        parquet_files = sorted(self.data_dir.rglob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

        ep_files = sorted((self.meta_dir / "episodes").rglob("*.parquet"))
        ep_dfs = [pd.read_parquet(f) for f in ep_files]
        self.episodes_df = pd.concat(ep_dfs, ignore_index=True)

        return self

    def get_states(self) -> np.ndarray:
        return np.array(self.df["observation.state"].tolist())

    def get_actions(self) -> np.ndarray:
        return np.array(self.df["action"].tolist())

    def get_episode_indices(self) -> np.ndarray:
        return self.df["episode_index"].values

    def get_timestamps(self) -> np.ndarray:
        return self.df["timestamp"].values

    def get_frame_indices(self) -> np.ndarray:
        return self.df["frame_index"].values


# ─── S1: 突变检测 ───────────────────────────────────────────────────────────

def s1_sudden_change_detection(
    signals: np.ndarray,
    config: CleaningConfig,
    report: CleaningReport,
    dim_names: Optional[list[str]] = None,
) -> np.ndarray:
    """
    检测突变帧并用平滑值替代（非丢弃）。
    流程: 中值滤波 → SG平滑 → 计算残差/加速度/急动度 → 联合阈值判定 → 替代为平滑值。
    返回: bool mask (True=正常, False=异常)。
    """
    N, D = signals.shape
    valid = np.ones(N, dtype=bool)

    for d in range(D):
        raw = signals[:, d]  # view, 修改会直接影响 signals

        # 级联中值滤波
        med_ksize = min(config.s1_medfilt_kernel, N - 1 if N % 2 == 0 else N)
        if med_ksize < 3:
            med_ksize = 3
        if med_ksize % 2 == 0:
            med_ksize -= 1
        smoothed = medfilt(raw, kernel_size=med_ksize)
        # SG平滑
        sg_window = min(config.s1_sg_window, N - 1 if N % 2 == 0 else N)
        if sg_window < 3:
            sg_window = 3
        if sg_window % 2 == 0:
            sg_window -= 1
        if N > sg_window:
            smoothed = savgol_filter(smoothed, sg_window, config.s1_sg_order)

        # 残差
        residual = np.abs(raw - smoothed)

        # 速度、加速度、急动度（手动填充保持长度N）
        v = np.diff(smoothed, prepend=smoothed[0])
        a = np.diff(v, prepend=v[0])
        j = np.diff(a, prepend=a[0])
        accel = np.abs(a)
        jerk = np.abs(j)

        # 阈值
        r_thresh = residual.mean() + config.s1_residual_std_mult * residual.std()
        a_thresh = accel.mean() + config.s1_accel_std_mult * accel.std()
        j_thresh = jerk.mean() + config.s1_jerk_std_mult * jerk.std()

        # 联合判定: 残差超阈值 AND (加速度超阈值 OR 急动度超阈值)
        flagged = (residual > r_thresh) & ((accel > a_thresh) | (jerk > j_thresh))

        n_flagged = flagged.sum()
        if n_flagged > 0:
            raw[flagged] = smoothed[flagged]  # 用平滑值替代，而非丢弃
            name = dim_names[d] if dim_names else f"dim_{d}"
            report.add("S1", f"{name}: smoothed {n_flagged} frames", {"smoothed_frames": int(n_flagged)})

    report.add("S1", f"Total smoothed: {(~valid).sum()} / {N} frames")
    return valid


# ─── S2: 状态-动作趋势对齐 ─────────────────────────────────────────────────

def s2_state_action_trend_alignment(
    states: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    config: CleaningConfig,
    report: CleaningReport,
    action_is_next_state: bool = True,
    dim_names: Optional[list[str]] = None,
) -> np.ndarray:
    """
    按 episode 检测状态-动作时间对齐质量。
    互相关估计最优延迟 → 计算方向一致性(DA) → 低于阈值则丢弃整段。
    返回: episode-level bool mask (True=保留, False=丢弃)。
    """
    ep_ids = np.unique(episode_indices)
    keep_ep = np.ones(len(ep_ids), dtype=bool)

    for ep_local_idx, ep_id in enumerate(ep_ids):
        mask = episode_indices == ep_id
        s_ep = states[mask]
        a_ep = actions[mask]

        if len(s_ep) < config.s2_sg_window + 5:
            continue

        da_list = []
        D = s_ep.shape[1]
        for d in range(D):
            s_smooth = savgol_filter(s_ep[:, d], config.s2_sg_window, config.s2_sg_order)
            a_smooth = savgol_filter(a_ep[:, d], config.s2_sg_window, config.s2_sg_order)

            # 互相关估计延迟
            corr = np.correlate(s_smooth - s_smooth.mean(), a_smooth - a_smooth.mean(), mode="same")
            mid = len(corr) // 2
            lag = np.argmax(corr[mid - config.s2_max_lag: mid + config.s2_max_lag + 1]) - config.s2_max_lag

            # 对齐后的方向一致性
            ds = np.diff(s_smooth)
            da_aligned = np.diff(a_smooth)
            if lag > 0:
                da_aligned = da_aligned[lag:]
                ds = ds[:len(da_aligned)]
            elif lag < 0:
                ds = ds[-lag:]
                da_aligned = da_aligned[:len(ds)]

            min_len = min(len(ds), len(da_aligned))
            ds, da_aligned = ds[:min_len], da_aligned[:min_len]

            same_sign = (ds * da_aligned) >= 0
            non_zero = (np.abs(ds) > 1e-10) | (np.abs(da_aligned) > 1e-10)
            if non_zero.sum() > 0:
                da_val = same_sign[non_zero].mean()
            else:
                da_val = 1.0
            da_list.append(da_val)

        min_da = min(da_list)
        if min_da < config.s2_da_threshold:
            keep_ep[ep_local_idx] = False
            report.add("S2", f"Episode {ep_id}: DA={min_da:.3f} < {config.s2_da_threshold}, discarded",
                       {"dropped_episodes": 1})

    n_dropped = len(ep_ids) - keep_ep.sum()
    report.add("S2", f"Episodes kept: {keep_ep.sum()} / {len(ep_ids)} (dropped {n_dropped})",
               {"dropped_episodes": int(n_dropped)})

    # 扩展回 frame-level mask
    frame_keep = np.ones(len(states), dtype=bool)
    for ep_local_idx, ep_id in enumerate(ep_ids):
        if not keep_ep[ep_local_idx]:
            frame_keep[episode_indices == ep_id] = False

    return frame_keep


# ─── S3: 极值过滤 ──────────────────────────────────────────────────────────

def s3_extreme_value_filtering(
    signals: np.ndarray,
    config: CleaningConfig,
    report: CleaningReport,
    dim_names: Optional[list[str]] = None,
) -> np.ndarray:
    """
    基于 Q1/Q99 的极值裁剪（非丢弃）。
    超限帧被裁剪到边界值。
    返回: frame-level bool mask。
    """
    N, D = signals.shape
    valid = np.ones(N, dtype=bool)

    for d in range(D):
        col = signals[:, d]
        q01 = np.percentile(col, 1)
        q99 = np.percentile(col, 99)

        name = dim_names[d] if dim_names else f"dim_{d}"

        # 夹爪豁免
        if config.s3_gripper_exempt and dim_names and "gripper" in dim_names[d].lower():
            report.add("S3", f"{name}: exempted (gripper)")
            continue

        lower = q01 - config.s3_alpha * (q99 - q01)
        upper = q99 + config.s3_alpha * (q99 - q01)

        flagged = (col < lower) | (col > upper)
        if flagged.sum() > 0:
            col[flagged] = np.clip(col[flagged], lower, upper)  # 裁剪到边界，而非丢弃
            report.add("S3", f"{name}: {flagged.sum()} frames clipped to [{lower:.4f}, {upper:.4f}]",
                       {"clipped_frames": int(flagged.sum())})

    report.add("S3", f"Total clipped: {(~valid).sum()} / {N} frames")
    return valid


# ─── S4: 正运动学一致性（无URDF时采用统计代理校验） ─────────────────────────

def s4_kinematic_consistency(
    states: np.ndarray,
    config: CleaningConfig,
    report: CleaningReport,
    dim_names: Optional[list[str]] = None,
) -> np.ndarray:
    """
    阶段1: 检测关节与末端位姿之间的异常关系。
    无 URDF 时，通过关节差分的协方差检测异常运动模式。
    返回: frame-level bool mask。
    """
    N, D = states.shape
    valid = np.ones(N, dtype=bool)

    # 计算关节速度（一阶差分）
    vel = np.diff(states, axis=0)
    vel_mag = np.linalg.norm(vel, axis=1)

    # 检测速度突变的帧（可能是 FK 不一致）
    vel_change = np.abs(np.diff(vel_mag, prepend=vel_mag[0]))
    thresh = vel_change.mean() + config.s4_pos_tolerance * vel_change.std()
    flagged = vel_change > thresh

    if flagged.sum() > 0:
        valid[1:][flagged] = False
        report.add("S4", f"FK anomaly: {flagged.sum()} frames with abnormal velocity change",
                   {"dropped_frames": int(flagged.sum())})

    report.add("S4", f"Total flagged: {(~valid).sum()} / {N} frames", {"dropped_frames": int((~valid).sum())})
    return valid


# ─── S5: 基座/末端方向对齐（单数据集: 统一符号约定） ────────────────────────

def s5_orientation_alignment(
    states: np.ndarray,
    report: CleaningReport,
) -> np.ndarray:
    """
    对单个数据集，统一关节方向符号约定。
    检测第3关节（典型负向范围 [-2, 0]）是否需要取反。
    返回: 对齐后的 states (新数组)。
    """
    aligned = states.copy()
    D = aligned.shape[1]
    corrections = []

    for d in range(D):
        col = aligned[:, d]
        # 检测是否大部分为负值且绝对值偏大 → 可能需要符号取反
        neg_ratio = (col < 0).mean()
        if neg_ratio > 0.8 and col.min() < -1.0:
            aligned[:, d] = -col
            corrections.append(f"dim_{d} flipped sign")
            report.add("S5", f"dim_{d}: flipped sign (neg_ratio={neg_ratio:.2f})")

    if not corrections:
        report.add("S5", "No orientation correction applied")
    return aligned


# ─── C1: 指令一致性（VLM代理检查: 规则+统计验证） ───────────────────────────

def c1_instruction_consistency(
    dataset: LeRobotDataset,
    report: CleaningReport,
) -> np.ndarray:
    """
    检查指令-演示语义一致性。
    无 VLM 时，使用统计验证: 检查同一 task_index 的指令是否语义一致，
    以及不同 task_index 的状态分布是否合理区分。
    返回: frame-level bool mask (True=通过)。
    """
    df = dataset.df
    tasks_df = pd.read_parquet(dataset.meta_dir / "tasks.parquet")
    task_texts = list(tasks_df.index)

    states = dataset.get_states()
    task_indices = df["task_index"].values
    valid = np.ones(len(df), dtype=bool)

    # 检查: 同一指令下状态分布是否合理（无极端离群 episode）
    for task_idx in df["task_index"].unique():
        ep_in_task = df[df["task_index"] == task_idx]["episode_index"].unique()
        if len(ep_in_task) < 2:
            continue

        ep_states = {}
        for ep in ep_in_task:
            mask = df["episode_index"] == ep
            ep_states[ep] = states[mask].mean(axis=0)

        ep_means = np.array(list(ep_states.values()))
        global_mean = ep_means.mean(axis=0)
        deviations = np.linalg.norm(ep_means - global_mean, axis=1)
        dev_thresh = deviations.mean() + 2 * deviations.std()
        outlier_eps = [list(ep_states.keys())[i] for i, d in enumerate(deviations) if d > dev_thresh]

        if outlier_eps:
            report.add("C1", f"Task {task_idx}: {outlier_eps} episodes deviate from task norm",
                       {"dropped_episodes": len(outlier_eps)})
            for ep in outlier_eps:
                valid[df["episode_index"] == ep] = False

    report.add("C1", f"Total flagged: {(~valid).sum()} / {len(valid)} frames")
    return valid


# ─── C2: 视频-状态一致性（统计代理：验证关节与时间戳的单调性） ─────────────

def c2_video_state_consistency(
    states: np.ndarray,
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    report: CleaningReport,
) -> np.ndarray:
    """
    无渲染环境时，通过时间戳单调性和关节变化率验证一致性。
    返回: frame-level bool mask。
    """
    valid = np.ones(len(states), dtype=bool)

    for ep_id in np.unique(episode_indices):
        mask = episode_indices == ep_id
        ts = timestamps[mask]
        st = states[mask]

        # 检查时间戳是否单调递增
        ts_diff = np.diff(ts)
        if (ts_diff < 0).any():
            n_bad = (ts_diff < 0).sum()
            report.add("C2", f"Episode {ep_id}: {n_bad} non-monotonic timestamps",
                       {"dropped_frames": int(n_bad)})
            # 标记非单调点
            bad_idx = np.where(mask)[0][1:][ts_diff < 0]
            valid[bad_idx] = False

        # 检查长时间无变化（可能是视频冻结）
        state_change = np.linalg.norm(np.diff(st, axis=0), axis=1)
        frozen = state_change < 1e-10
        if frozen.sum() > 5:
            report.add("C2", f"Episode {ep_id}: {frozen.sum()} frozen frames detected",
                       {"dropped_frames": int(frozen.sum())})

    report.add("C2", f"Total flagged: {(~valid).sum()} / {len(valid)} frames")
    return valid


# ─── C3: 视频质量过滤（统计代理：检查帧索引连续性和重复帧） ─────────────────

def c3_video_quality_filtering(
    frame_indices: np.ndarray,
    episode_indices: np.ndarray,
    states: np.ndarray,
    report: CleaningReport,
) -> np.ndarray:
    """
    检测帧索引跳变、静态段、帧重复等质量问题。
    返回: frame-level bool mask。
    """
    valid = np.ones(len(frame_indices), dtype=bool)

    for ep_id in np.unique(episode_indices):
        mask = episode_indices == ep_id
        fi = frame_indices[mask]
        st = states[mask]

        # 帧索引跳变
        fi_diff = np.diff(fi)
        jump = fi_diff > 1
        if jump.sum() > 0:
            n_jump = jump.sum()
            report.add("C3", f"Episode {ep_id}: {n_jump} frame index jumps",
                       {"dropped_frames": int(n_jump)})

        # 尾部静态段检测（episode 末尾无动作的区域）
        state_change = np.linalg.norm(np.diff(st, axis=0), axis=1)
        if len(state_change) > 10:
            # 从末尾向前找连续静态段
            static_tail = 0
            for i in range(len(state_change) - 1, -1, -1):
                if state_change[i] < 1e-10:
                    static_tail += 1
                else:
                    break
            if static_tail > 10:
                tail_start_idx = np.where(mask)[0][-static_tail:]
                valid[tail_start_idx] = False
                report.add("C3", f"Episode {ep_id}: {static_tail} tail static frames removed",
                           {"dropped_frames": int(static_tail)})

    report.add("C3", f"Total flagged: {(~valid).sum()} / {len(valid)} frames")
    return valid


# ─── 主流水线 ───────────────────────────────────────────────────────────────

class DataCleaningPipeline:
    """数据清洗管线主控器"""

    STAGE_NAMES = ["S1_SuddenChange", "S2_TrendAlignment", "S3_ExtremeValue",
                    "S4_KinematicConsistency", "S5_OrientationAlignment",
                    "C1_InstructionConsistency", "C2_VideoStateConsistency",
                    "C3_VideoQuality"]

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.report = CleaningReport()
        self.dim_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        self.dataset: Optional[LeRobotDataset] = None
        self._states: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._frame_mask: Optional[np.ndarray] = None

    def run(self, data_root: str, stages: Optional[list[str]] = None):
        """执行清洗管线"""
        if stages is None:
            stages = self.STAGE_NAMES

        self.dataset = LeRobotDataset(data_root).load()
        self._states = self.dataset.get_states()
        self._actions = self.dataset.get_actions()
        self._frame_mask = np.ones(len(self._states), dtype=bool)

        print(f"Dataset: {self.dataset.robot_type}, {self.dataset.total_frames} frames, "
              f"{self.dataset.total_episodes} episodes")
        print(f"State/action dim: {self._states.shape[1]}")

        stage_map = {
            "S1_SuddenChange": self._run_s1,
            "S2_TrendAlignment": self._run_s2,
            "S3_ExtremeValue": self._run_s3,
            "S4_KinematicConsistency": self._run_s4,
            "S5_OrientationAlignment": self._run_s5,
            "C1_InstructionConsistency": self._run_c1,
            "C2_VideoStateConsistency": self._run_c2,
            "C3_VideoQuality": self._run_c3,
        }

        for stage_name in stages:
            if stage_name in stage_map:
                print(f"\n{'='*60}")
                print(f"Stage: {stage_name}")
                print(f"{'='*60}")
                self._frame_mask = stage_map[stage_name](self._frame_mask)

        # 最终统计
        n_kept = self._frame_mask.sum()
        n_total = len(self._frame_mask)
        print(f"\n{'='*60}")
        print(f"Cleanup complete: {n_kept}/{n_total} frames kept ({n_kept/n_total*100:.1f}%)")

        # 清洗后 frame → episode 映射
        kept_eps = set(self.dataset.df.loc[self._frame_mask, "episode_index"])
        n_ep_total = self.dataset.total_episodes
        print(f"Episodes with any frames remaining: {len(kept_eps)}/{n_ep_total}")

        return self._frame_mask

    def get_clean_data(self) -> pd.DataFrame:
        if self._frame_mask is None:
            raise RuntimeError("Run pipeline first")
        return self.dataset.df.iloc[self._frame_mask].copy()

    def save_report(self, path: str):
        Path(path).write_text(json.dumps(self.report.stage_logs, indent=2, default=str))
        print(f"\nReport saved to {path}")

    def _apply_mask(self, mask: np.ndarray, name: str) -> np.ndarray:
        new_mask = self._frame_mask.copy()
        new_mask[~mask] = False
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"  → {name}: filtered {dropped} frames")
        return new_mask

    def _run_s1(self, current_mask: np.ndarray) -> np.ndarray:
        subset = self._states[current_mask]
        if len(subset) == 0:
            return current_mask
        s1_sudden_change_detection(subset, self.config, self.report, self.dim_names)
        self._states[current_mask] = subset
        return current_mask

    def _run_s2(self, current_mask: np.ndarray) -> np.ndarray:
        states, actions = self._states[current_mask], self._actions[current_mask]
        eps = self.dataset.df.loc[current_mask, "episode_index"].values
        if len(states) == 0:
            return current_mask
        stage_valid = s2_state_action_trend_alignment(
            states, actions, eps, self.config, self.report, dim_names=self.dim_names)
        full_valid = np.ones(len(current_mask), dtype=bool)
        full_valid[current_mask] = stage_valid
        return self._apply_mask(full_valid, "S2")

    def _run_s3(self, current_mask: np.ndarray) -> np.ndarray:
        subset = self._states[current_mask]
        if len(subset) == 0:
            return current_mask
        s3_extreme_value_filtering(subset, self.config, self.report, self.dim_names)
        self._states[current_mask] = subset
        return current_mask

    def _run_s4(self, current_mask: np.ndarray) -> np.ndarray:
        states = self._states[current_mask]
        if len(states) == 0:
            return current_mask
        stage_valid = s4_kinematic_consistency(states, self.config, self.report, self.dim_names)
        full_valid = np.ones(len(current_mask), dtype=bool)
        full_valid[current_mask] = stage_valid
        return self._apply_mask(full_valid, "S4")

    def _run_s5(self, current_mask: np.ndarray) -> np.ndarray:
        # S5 修改数据本身（不过滤）
        aligned = s5_orientation_alignment(self._states, self.report)
        self._states = aligned
        print("  → S5: orientation alignment applied to states")
        return current_mask

    def _run_c1(self, current_mask: np.ndarray) -> np.ndarray:
        stage_valid = c1_instruction_consistency(self.dataset, self.report)
        # 只对当前仍有效的帧做检查
        stage_valid[~current_mask] = True
        new_mask = current_mask.copy()
        new_mask[~stage_valid] = False
        dropped = current_mask.sum() - new_mask.sum()
        print(f"  → C1: filtered {dropped} frames")
        return new_mask

    def _run_c2(self, current_mask: np.ndarray) -> np.ndarray:
        states = self._states[current_mask]
        ts = self.dataset.df.loc[current_mask, "timestamp"].values
        eps = self.dataset.df.loc[current_mask, "episode_index"].values
        if len(states) == 0:
            return current_mask
        stage_valid = c2_video_state_consistency(states, ts, eps, self.report)
        full_valid = np.ones(len(current_mask), dtype=bool)
        full_valid[current_mask] = stage_valid
        return self._apply_mask(full_valid, "C2")

    def _run_c3(self, current_mask: np.ndarray) -> np.ndarray:
        fi = self.dataset.df.loc[current_mask, "frame_index"].values
        eps = self.dataset.df.loc[current_mask, "episode_index"].values
        states = self._states[current_mask]
        if len(states) == 0:
            return current_mask
        stage_valid = c3_video_quality_filtering(fi, eps, states, self.report)
        full_valid = np.ones(len(current_mask), dtype=bool)
        full_valid[current_mask] = stage_valid
        return self._apply_mask(full_valid, "C3")


# ─── 主入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("data_root", nargs="?", default=None,
                        help="Path to dataset root")
    parser.add_argument("--stages", nargs="+", default=None,
                        choices=["S1_SuddenChange", "S2_TrendAlignment", "S3_ExtremeValue",
                                 "S4_KinematicConsistency", "S5_OrientationAlignment",
                                 "C1_InstructionConsistency", "C2_VideoStateConsistency",
                                 "C3_VideoQuality"],
                        help="Stages to run (default: S1 + S3)")
    args = parser.parse_args()

    data_root = args.data_root if args.data_root else \
        "/home/alex/workspace/Code/数据优化算子/example_data/pick_banana_100_newTable_1_offset_state"

    config = CleaningConfig(
        s1_residual_std_mult=3.0,
        s1_accel_std_mult=3.0,
        s1_jerk_std_mult=3.0,
        s2_da_threshold=0.6,
        s3_alpha=1.5,
        s3_gripper_exempt=True,
    )

    if args.stages is None:
        args.stages = ["S1_SuddenChange", "S3_ExtremeValue"]
    pipeline = DataCleaningPipeline(config)
    mask = pipeline.run(data_root, stages=args.stages)

    clean_df = pipeline.get_clean_data()
    out_path = Path(data_root) / ".." / "pick_banana_100_newTable_1_offset_state_cleaned"
    out_path.mkdir(exist_ok=True)
    clean_df.to_parquet(out_path / "cleaned_data.parquet")
    print(f"\nCleaned data saved to {out_path / 'cleaned_data.parquet'}")

    pipeline.save_report(str(out_path / "cleaning_report.json"))
