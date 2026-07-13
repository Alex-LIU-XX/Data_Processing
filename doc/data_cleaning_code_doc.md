# 数据清洗管线代码文档

**文件**: `/home/alex/workspace/Code/数据优化算子/data_cleaning_pipeline.py`

---

## 1. 架构总览

```
DataCleaningPipeline
├── S1_SuddenChange           # 突变检测
├── S2_TrendAlignment         # 状态-动作趋势对齐
├── S3_ExtremeValue           # 极值过滤
├── S4_KinematicConsistency   # 正运动学一致性(统计代理)
├── S5_OrientationAlignment   # 基座/末端方向对齐
├── C1_InstructionConsistency # 指令一致性
├── C2_VideoStateConsistency  # 视频-状态一致性
└── C3_VideoQuality           # 视频质量过滤
```

**核心设计模式**: 每个 S/C 阶段是一个独立函数,接收 `(数据, 配置, 报告)` 并返回 bool mask。
`DataCleaningPipeline` 类负责串联:每个阶段通过 `_run_sX` 包装方法管理 mask 传播,逐步收紧 mask。

**数据流**:
```
原始 DataFrame ──→ S1 mask ──→ S2 mask ──→ ... ──→ C3 mask ──→ Clean DataFrame
     ↓                 ↓           ↓                    ↓
  states_np       仅 S1 保留     仅 S1+S2            最终保留
  actions_np      的帧上处理     保留帧上处理          的帧
```

---

## 2. 核心类

### 2.1 `CleaningConfig` — 阈值配置容器

```python
@dataclass
class CleaningConfig:
    # S1: 突变检测
    s1_residual_std_mult: float = 3.0    # 残差阈值 = mean + mult * std
    s1_accel_std_mult: float = 3.0       # 加速度阈值 = mean + mult * std
    s1_jerk_std_mult: float = 3.0        # 急动度阈值 = mean + mult * std
    s1_medfilt_kernel: int = 5           # 中值滤波核大小
    s1_sg_window: int = 11               # Savitzky-Golay 窗口
    s1_sg_order: int = 3                 # SG 多项式阶数

    # S2: 趋势对齐
    s2_da_threshold: float = 0.6         # 方向一致性阈值
    s2_sg_window: int = 11              # SG 平滑窗口
    s2_sg_order: int = 3                 # SG 多项式阶数
    s2_max_lag: int = 10                 # 互相关最大搜索延迟

    # S3: 极值过滤
    s3_alpha: float = 1.5               # 极值 band 扩展系数
    s3_gripper_exempt: bool = True       # 夹爪豁免

    # S4: FK一致性（暂无URDF时使用统计校验）
    s4_pos_tolerance: float = 0.1        # 位置速度异常检测阈值倍数
    s4_rot_tolerance: float = 0.2        # 姿态速度异常检测阈值倍数(预留)

    # 综合
    episode_discard_on_corrupt: bool = False
```

每个参数设计为 **per-dataset 可覆盖**,通过在实例化时传参实现。

### 2.2 `LeRobotDataset` — 数据加载器

```python
class LeRobotDataset:
    def __init__(self, root: str):
        # 从 meta/info.json 读取元信息(robot_type, fps, total_episodes, total_frames)
        # 从 meta/action_patch_info.json 读取 action 构造策略
        # 从 meta/stats.json 读取预计算的统计量

    def load(self):
        # 递归扫描 data/**/*.parquet 并合并 → self.df
        # 递归扫描 meta/episodes/**/*.parquet 并合并 → self.episodes_df
```

核心方法:
- `get_states()` → `np.ndarray` shape (N, D)
- `get_actions()` → `np.ndarray` shape (N, D)
- `get_episode_indices()` → `np.ndarray` shape (N,)
- `get_timestamps()` → `np.ndarray` shape (N,)
- `get_frame_indices()` → `np.ndarray` shape (N,)

### 2.3 `CleaningReport` — 日志记录器

```python
@dataclass
class CleaningReport:
    stage_logs: dict = field(default_factory=dict)
    # 结构: {"S1": [{"msg": "...", "detail": {...}}, ...], "S2": [...]}

    def add(self, stage, msg, detail=None):
    def summary(self) -> str:
```

---

## 3. 算子实现详解

### S1: 突变检测 — `s1_sudden_change_detection`

**目的**: 检测并标记物理碰撞、传感器异常引起的信号瞬态突变。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `signals: np.ndarray` shape (N, D), `config`, `report`, `dim_names` |
| 输出 | `valid: np.ndarray` shape (N,) bool, True=正常帧 |
| 维度处理 | 逐维度独立处理 |

#### 算法步骤

```
对每个维度 d in [0, D):
  ├─ 步骤1: 级联中值滤波
  │    raw = signals[:, d]
  │    kernel = min(config.s1_medfilt_kernel, N-1 若 N 为偶数否则 N)
  │    kernel 调整为 ≥3 的奇数
  │    smoothed = medfilt(raw, kernel_size=kernel)
  │
  ├─ 步骤2: Savitzky-Golay 平滑
  │    window = min(config.s1_sg_window, N-1 若 N 为偶数否则 N)
  │    window 调整为 ≥3 的奇数
  │    if N > window: smoothed = savgol_filter(smoothed, window, order=3)
  │
  ├─ 步骤3: 计算三信号
  │    residual = |raw - smoothed|              # 残差: 原始与平滑的偏差
  │    v = diff(smoothed, prepend=smoothed[0])  # 速度: 一阶差分
  │    a = diff(v, prepend=v[0])                # 加速度: 二阶差分
  │    j = diff(a, prepend=a[0])                # 急动度: 三阶差分
  │
  ├─ 步骤4: 动态阈值计算
  │    r_thresh = mean(residual) + mult * std(residual)
  │    a_thresh = mean(|a|) + mult * std(|a|)
  │    j_thresh = mean(|j|) + mult * std(|j|)
  │
  └─ 步骤5: 联合判定
       flagged = (residual > r_thresh) & ((|a| > a_thresh) | (|j| > j_thresh))
       # 残差超阈值(排除慢漂移)
       # AND (加速度超阈值 OR 急动度超阈值)
       # 两个条件同时满足才标记,降低误报
       valid[flagged] = False
```

#### 关键设计决策

1. **联合判定而非单阈值**: 仅当残差和加速度/急动度同时超标时才标记,避免将平滑运动(如快速接近物体)误判为突变。
2. **动态阈值**: 每维数据自适应计算 mean+mult*std,无需手动指定物理单位对应的阈值。
3. **核自适应**: 当信号长度短于滤波窗口时自动缩小窗口,防止短 episode 报错。
4. **N 长度保持**: 用 `prepend=smoothed[0]` 计算差分,确保输出长度始终等于 N。

#### 边缘情况处理

- 信号长度 N 较小时自动收缩滤波窗口,保证滤波器可执行。
- 常量信号(如夹爪保持不动): std ≈ 0, 不会触发标记。

#### 运行结果(香蕉数据集)

```
joint_1:  44 frames flagged
joint_2:   2 frames flagged
joint_3:  89 frames flagged
joint_4:  35 frames flagged
joint_5:  58 frames flagged
joint_6:  16 frames flagged
gripper:  36 frames flagged
Total:   280 / 12,209 frames (2.3%)
```

---

### S2: 状态-动作趋势对齐 — `s2_state_action_trend_alignment`

**目的**: 校正异步时钟、数据包丢失引起的状态-动作时间错位。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `states` (N,D), `actions` (N,D), `episode_indices` (N,), `config`, `report` |
| 输出 | `frame_keep: np.ndarray` shape (N,) bool |
| 作用粒度 | 按 episode 判定,整段丢弃 |

#### 算法步骤

```
对每个 episode:
  ├─ 提取该 episode 的状态和动作子序列
  │    s_ep = states[episode_indices == ep_id]
  │    a_ep = actions[episode_indices == ep_id]
  │
  ├─ 对每个维度 d:
  │    ├─ SG平滑(窗口 config.s2_sg_window, 阶 config.s2_sg_order)
  │    │    s_smooth = savgol_filter(s_ep[:, d], s2_sg_window, s2_sg_order)
  │    │    a_smooth = savgol_filter(a_ep[:, d], s2_sg_window, s2_sg_order)
  │    │
  │    ├─ 互相关估计延迟
  │    │    corr = correlate(s_smooth - μ, a_smooth - μ, mode="same")
  │    │    mid = len(corr) // 2
  │    │    lag = argmax(corr[mid-max_lag : mid+max_lag+1]) - max_lag
  │    │
  │    ├─ 对齐后计算方向一致性
  │    │    ds = diff(s_smooth)           # 状态变化方向
  │    │    da = diff(a_smooth)           # 动作变化方向
  │    │    if lag > 0: da = da[lag:]
  │    │               ds = ds[:len(da)]
  │    │    if lag < 0: ds = ds[-lag:]
  │    │               da = da[:len(ds)]
  │    │    # 截断至等长
  │    │    same_sign = (ds * da) >= 0
  │    │    non_zero = (|ds| > 1e-10) | (|da| > 1e-10)
  │    │    DA_d = mean(same_sign[non_zero])  # 方向一致性指标
  │    │
  │    └─ 收集 DA_d
  │
  ├─ min_DA = min(所有维度的 DA_d)
  │
  └─ if min_DA < config.s2_da_threshold:
       episode 被标记丢弃 (全 True → False)
```

#### 方向一致性(DA)的物理含义

DA 衡量的是"状态变化方向与动作变化方向是否一致"的比例。例如:
- 关节角度从 0.5 变到 0.6 (状态变化 d_state > 0)
- 动作输出从 0.55 变到 0.65 (动作变化 d_action > 0)
- 两者方向一致 → 计入 same_sign

DA = 0.97 意味着 97% 的非零变化帧方向一致,3% 可能是量化噪声或微小反冲。

#### 关键设计决策

1. **Episode 级判定**: 时间对齐问题通常是系统性的(如整段数据采集时钟差),整段丢弃比逐帧修正更合理。
2. **取最小 DA**: 只要有一个维度的对齐质量差就丢弃整段,因为任何一个关节不同步都意味着数据不可用。
3. **SG 平滑预处理**: 消除高频噪声对互相关和 DA 的影响。

#### 边缘情况处理

- 短 episode (N < window+5): 跳过检查,直接保留。
- 纯静态 episode: `non_zero` 接近 0,直接设 DA=1.0。

#### 运行结果(香蕉数据集)

```
100/100 episodes kept (DA > 0.6 for all)
```

说明此数据采集时钟同步良好,无需丢弃。

---

### S3: 极值过滤 — `s3_extreme_value_filtering`

**目的**: 去除超出合理物理范围的帧,防止训练归一化时被异常值扭曲。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `signals` (N,D), `config`, `report` |
| 输出 | `valid` (N,) bool |

#### 算法步骤

```
对每个维度 d:
  ├─ col = signals[:, d]
  ├─ 检查是否夹爪维度且配置豁免
  │    if config.s3_gripper_exempt and dim_names[d] 含 "gripper":
  │       跳过此维度 (因为夹爪呈双峰分布,极值正常)
  │
  ├─ 计算百分位
  │    q01 = percentile(col, 1)
  │    q99 = percentile(col, 99)
  │
  ├─ 定义合理区间
  │    spread = q99 - q01                    # 核心分布宽度
  │    lower = q01 - alpha * spread          # 下界
  │    upper = q99 + alpha * spread          # 上界
  │
  └─ 标记异常帧
       flagged = (col < lower) | (col > upper)
       valid[flagged] = False
```

#### 参数 α=1.5 的含义

```
alpha=1.5 时:
  band 宽度 = 2 * 1.5 * (q99-q01) + (q99-q01) = 4 * (q99-q01)
  即 band 总宽度 ≈ 4倍核心区间宽度
```

这个值足够宽松,只有真正的离群值才会被过滤。

#### 夹爪豁免的理由

论文指出夹爪呈现**双峰分布**(全开≈0,全闭≈1),Q1-Q99 区间极小的情况下,按同样标准会误杀正常帧。

#### 运行结果(香蕉数据集)

```
所有维度均无极值帧被标记
夹爪维度已豁免
```

与预期一致,此数据集无明显记录异常。

---

### S4: 正运动学一致性 — `s4_kinematic_consistency`

**目的**: 通过关节空间速度变化率检测异常运动模式,作为 FK 校验的统计替代方案。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `states` (N,D), `config`, `report` |
| 输出 | `valid` (N,) bool |

#### 当前实现(统计代理)

```
├─ 计算关节空间速度
│    vel = diff(states, axis=0) → (N-1, D)
│    vel_mag = ||vel||_2           → (N-1,)  逐帧速率幅值
│
├─ 速度变化率(急动度代理)
│    vel_change = |diff(vel_mag, prepend=vel_mag[0])|  → (N-1,)
│
├─ 阈值
│    thresh = mean(vel_change) + s4_pos_tolerance * std(vel_change)
│
└─ 标记异常
     flagged = vel_change > thresh
     valid[1:][flagged] = False
```

#### 为什么用速度变化率代理 FK?

理想的 FK 校验是:
1. 读取关节角度 → 通过 URDF/Pinocchio 计算 FK → 得到理论末端位姿
2. 比较理论位姿与记录位姿的差异
3. 若差异超过阈值,说明数据存在不一致

在缺乏 URDF 时,**速度变化率检测**作为代理:
- 真实 FK 不一致常表现为"关节角度变化小但末端突变"(不可实现)
- 速度变化率异常可以部分捕获这类情况

#### 局限性

| 问题 | 影响 |
|------|------|
| 无法区分 FK 不一致和正常快速运动 | 误报率较高 |
| 无法修正 TCP 偏移/旋转表示 | 只能丢弃,不能修正 |
| 对慢速漂移不敏感 | 漏报 |

**论文方案**使用 Pinocchio 库 + URDF 进行真正 FK 校验,并具备自动修正能力(自动调整 TCP 偏移、旋转表示等)。

#### 运行结果(香蕉数据集)

```
2,226 frames (18.2%) flagged as FK anomaly
```

这个比例明显偏高,说明统计代理过于激进。大部分被标记的帧很可能是正常的快速运动而非真正的 FK 不一致。获取 URDF 后应替换为真正的 FK 校验。

---

### S5: 方向对齐 — `s5_orientation_alignment`

**目的**: 统一不同数据集间因坐标系定义不同导致的关节符号差异。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `states` (N,D), `report` |
| 输出 | `aligned` (N,D) — 修改后的状态数组 |

注意: S5 **修改数据本身**,而非生成过滤 mask。

#### 算法步骤

```
对每个维度 d:
  col = states[:, d]
  neg_ratio = (col < 0).mean()     # 负值占比

  if neg_ratio > 0.8 and col.min() < -1.0:
      # 该维度的值几乎全为负且幅度大 → 可能是符号约定问题
      aligned[:, d] = -col          # 取反
      report.add("S5", f"dim_{d}: flipped sign")
```

#### 判定标准

- **neg_ratio > 0.8**: 80%+ 的值为负
- **col.min() < -1.0**: 负向幅度超过 1.0 rad (非噪声)

两个条件同时满足时触发符号取反,避免错误反转噪声信号。

#### 为什么需要方向对齐?

在不同数据集或采集系统中,同一关节可能使用相反的符号约定。例如:
- 数据集 A: 关节向外转为正
- 数据集 B: 关节向外转为负

这种不一致在跨本体训练时会导致冲突,模型无法学到一致的物理运动。

#### 运行结果(香蕉数据集)

```
dim_2 (joint_3): flipped sign (neg_ratio=0.99)
```

joint_3 原始范围 [-2.017, 0.001], 99% 为负值,取反后变为 [0, 2.017]。

---

### C1: 指令一致性 — `c1_instruction_consistency`

**目的**: 验证语言指令与演示内容语义一致。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `dataset: LeRobotDataset`, `report` |
| 输出 | `valid` (N,) bool |

#### 当前实现(统计代理)

```
├─ 读取 tasks.parquet 获取 task_index → 指令文本映射
│
├─ 对每个 task_index:
│    ├─ 找到该任务下的所有 episode
│    ├─ 计算每个 episode 的状态均值向量
│    │    ep_states[ep] = mean(states[episode==ep], axis=0)
│    │
│    ├─ 计算该任务下所有 episode 的全局均值
│    │    global_mean = mean(各episode均值, axis=0)
│    │
│    ├─ 计算每个 episode 到全局均值的偏差
│    │    deviation = ||ep_mean - global_mean||_2
│    │
│    └─ 标记偏差超过 2σ 的 episode
│         outlier = deviation > mean(deviation) + 2 * std(deviation)
│         valid[outlier_episodes] = False
│
└─ 返回 mask
```

#### 代理方案的假设

**假设**: 同一指令对应的演示在状态空间中应具有相似的分布。如果某个 episode 的状态分布显著偏离同任务的其它 episode,则可能是数据标记错误或采集异常。

#### 与论文方案的差距

论文使用 **三级 VLM 流水线**:
1. **分段**: 长视频分解为子任务级片段
2. **结构化推理**: VLM 被引导分析物体、动作语义、时序后给出判断
3. **多专家仲裁**: 多个 VLM 独立评估后投票

这是对语义的**直接验证**,而非基于分布的统计推断。

#### 运行结果(香蕉数据集)

```
0 frames filtered — 所有 episode 在同任务内分布一致
```

---

### C2: 视频-状态一致性 — `c2_video_state_consistency`

**目的**: 验证视频中机器人的视觉外观与记录的状态数据一致。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `states` (N,D), `timestamps` (N,), `episode_indices` (N,), `report` |
| 输出 | `valid` (N,) bool |

#### 当前实现(统计代理)

```
对每个 episode:
  ├─ 检查1: 时间戳单调性
  │    ts_diff = diff(timestamps[mask])
  │    if any(ts_diff < 0):
  │        标记非单调帧为异常
  │
  └─ 检查2: 关节冻结检测
       state_change = ||diff(states, axis=0)||_2
       frozen = state_change < 1e-10
       if frozen.sum() > 5:
           记录这些帧(但不自动丢弃)
```

#### 与论文方案的差距

论文使用 **URDF 渲染 + SAM3 分割 + IoU 计算**:
1. 用 URDF + 关节角度渲染机器人图像
2. SAM3 分割实际视频中的机器人 mask
3. 计算渲染 mask 与分割 mask 的 IoU
4. IoU < 阈值 → 排除或优化相机参数

这是**几何验证**,可以直接检测相机参数错误、遮挡等真实问题。

#### 运行结果(香蕉数据集)

```
0 frames filtered — 时间戳单调,无冻结帧
```

---

### C3: 视频质量过滤 — `c3_video_quality_filtering`

**目的**: 剔除视觉上无效的视频帧(黑帧、损坏、模糊、静态段)。

#### 输入/输出

| 项目 | 说明 |
|------|------|
| 输入 | `frame_indices` (N,), `episode_indices` (N,), `states` (N,D), `report` |
| 输出 | `valid` (N,) bool |

#### 当前实现(基于元数据而非像素)

```
对每个 episode:
  ├─ 检查1: 帧索引跳变 (仅记录,不丢弃)
  │    fi_diff = diff(frame_indices[mask])
  │    jump = fi_diff > 1
  │    记录跳变位置
  │    # frame_index 可能受 action offset 影响而非真正的视频丢帧
  │
  └─ 检查2: 尾部静态段检测
       state_change = ||diff(states, axis=0)||_2
       从末尾向前扫描连续静态帧:
         while state_change[i] < 1e-10:
             static_tail++
         break
       if static_tail > 10:
           标记尾部 static_tail 帧为异常
           # episode 末尾的冗余静态帧对训练无贡献
```

#### 与论文方案的差距

论文使用**像素级检测**:
- 黑帧检测: 像素值方差接近 0
- 损坏帧检测: 解码错误
- 模糊检测: Laplacian 方差
- 静态段检测: 联合视觉+状态变化量
- **关键帧保留**: 夹爪闭合事件等仍保留

当前实现仅使用元数据(帧索引 + 状态),无法检测像素级质量问题。

#### 运行结果(香蕉数据集)

```
0 frames filtered — 帧索引跳变仅在日志中记录,尾部无长静态段
所有 episode 均有帧索引跳变日志(正常: 因 action offset 导致)
```

---

## 4. 管线编排流程

### 4.1 串联逻辑

```python
class DataCleaningPipeline:
    STAGE_NAMES = ["S1_SuddenChange", "S2_TrendAlignment", "S3_ExtremeValue",
                    "S4_KinematicConsistency", "S5_OrientationAlignment",
                    "C1_InstructionConsistency", "C2_VideoStateConsistency",
                    "C3_VideoQuality"]

    def run(self, data_root: str, stages: Optional[list[str]] = None):
        # 1. 加载数据
        self.dataset = LeRobotDataset(data_root).load()
        self._states = self.dataset.get_states()
        self._actions = self.dataset.get_actions()
        self._frame_mask = np.ones(len(self._states), dtype=bool)

        # 2. 逐阶段执行（每个阶段通过 _run_sX 包装方法管理 mask）
        stage_map = {
            "S1_SuddenChange": self._run_s1,
            "S2_TrendAlignment": self._run_s2,
            # ...
        }
        for stage_name in stages:
            self._frame_mask = stage_map[stage_name](self._frame_mask)

        # 3. 返回最终 mask
        return self._frame_mask
```

### 4.2 Mask 传播机制

每个阶段函数只接收当前 `_frame_mask` 下仍有效的子集数据,返回子集上的 bool mask,通过 `_run_sX` 包装方法映射回全局:

```python
def _run_s1(self, current_mask: np.ndarray) -> np.ndarray:
    states = self._states[current_mask]          # 提取子集
    stage_valid = s1_sudden_change_detection(...) # 子集上运算
    full_valid = np.ones(len(current_mask), dtype=bool)
    full_valid[current_mask] = stage_valid        # 映射回全局
    return self._apply_mask(full_valid, "S1")     # 合并到主 mask

def _apply_mask(self, mask: np.ndarray, name: str) -> np.ndarray:
    new_mask = self._frame_mask.copy()
    new_mask[~mask] = False
    return new_mask
```

### 4.3 Mask 传播示意图

```
初始:   ┌──────────────────────────────────────────┐
        │ 1 1 1 1 1 1 1 1 1 1 ... (12209 帧)      │
S1 后:  │ 1 1 0 1 1 1 1 0 1 1 ... (11929 帧)      │
S2 后:  │ 1 1 0 1 1 1 1 0 1 1 ... (11929 帧, 无变化)│
S3 后:  │ 1 1 0 1 1 1 1 0 1 1 ... (11929 帧, 无变化)│
S4 后:  │ 1 0 0 1 0 1 0 0 1 0 ... (9703 帧)       │
S5 后:  │ 1 0 0 1 0 1 0 0 1 0 ... (数据修改, 无过滤)│
C1 后:  │ 1 0 0 1 0 1 0 0 1 0 ... (9703 帧, 无变化)│
C2 后:  │ 1 0 0 1 0 1 0 0 1 0 ... (9703 帧, 无变化)│
C3 后:  │ 1 0 0 1 0 1 0 0 1 0 ... (9703 帧)       │
        └──────────────────────────────────────────┘
```

S5 不修改 mask,直接更新 `self._states` 数据。

---

## 5. 运行结果验证

### 5.1 香蕉数据集的清洗结果

| 阶段 | 输入帧 | 输出帧 | 过滤帧 | 说明 |
|------|--------|--------|--------|------|
| S1 | 12,209 | 11,929 | 280 | 突变帧移除 |
| S2 | 11,929 | 11,929 | 0 | 所有 episode DA > 0.6 |
| S3 | 11,929 | 11,929 | 0 | 无极值帧 |
| S4 | 11,929 | 9,703 | 2,226 | 统计代理偏激进 |
| S5 | 9,703 | 9,703 | — | joint_3 符号取反(不丢帧) |
| C1 | 9,703 | 9,703 | 0 | 分布无异常 |
| C2 | 9,703 | 9,703 | 0 | 时间戳正常 |
| C3 | 9,703 | 9,703 | 0 | 无长静态段 |
| **总计** | **12,209** | **9,703** | **2,506 (20.5%)** | |

### 5.2 清洗报告 JSON 示例

```json
{
  "S1": [
    {"msg": "joint_1: flagged 44 frames", "detail": {"dropped_frames": 44}},
    {"msg": "joint_2: flagged 2 frames",  "detail": {"dropped_frames": 2}},
    {"msg": "joint_3: flagged 89 frames", "detail": {"dropped_frames": 89}},
    {"msg": "joint_4: flagged 33 frames", "detail": {"dropped_frames": 33}},
    {"msg": "joint_5: flagged 58 frames", "detail": {"dropped_frames": 58}},
    {"msg": "joint_6: flagged 16 frames", "detail": {"dropped_frames": 16}},
    {"msg": "gripper: flagged 36 frames", "detail": {"dropped_frames": 36}},
    {"msg": "Total flagged: 280 / 12209",  "detail": {"dropped_frames": 280}}
  ],
  "S2": [
    {"msg": "Episodes kept: 100 / 100", "detail": {"dropped_episodes": 0}}
  ],
  "S3": [
    {"msg": "gripper: exempted (gripper)"},
    {"msg": "Total flagged: 0 / 11929 frames", "detail": {"dropped_frames": 0}}
  ],
  "S5": [
    {"msg": "dim_2: flipped sign (neg_ratio=0.99)"}
  ],
  "C3": [
    {"msg": "Episode 0: 11 frame index jumps", "detail": {"dropped_frames": 0}},
    {"msg": "Episode 1: 14 frame index jumps", "detail": {"dropped_frames": 0}},
    ...
  ]
}
```

---

## 6. 使用方式

```bash
# 完整管线
python data_cleaning_pipeline.py <dataset_root>

# 仅运行指定阶段
python -c "
from data_cleaning_pipeline import DataCleaningPipeline
p = DataCleaningPipeline()
p.run('<dataset_root>', stages=['S1_SuddenChange', 'S3_ExtremeValue'])
"
```

```python
from data_cleaning_pipeline import DataCleaningPipeline, CleaningConfig

config = CleaningConfig(s1_residual_std_mult=2.5, s2_da_threshold=0.65)
pipeline = DataCleaningPipeline(config)
mask = pipeline.run("/path/to/dataset")
clean_df = pipeline.get_clean_data()
pipeline.save_report("/path/to/report.json")
```

### 主入口说明

`data_cleaning_pipeline.py` 的 `__main__` 块接受可选命令行参数作为数据集路径:

```bash
python data_cleaning_pipeline.py                                    # 使用默认 example_data 路径
python data_cleaning_pipeline.py /path/to/lerobot/v3.0/dataset      # 指定数据集
```

执行后自动:
1. 将清洗后数据保存为 `../{数据集名}_cleaned/cleaned_data.parquet`
2. 将清洗报告保存为 `../{数据集名}_cleaned/cleaning_report.json`

---

## 7. 待完善项

| 阶段 | 当前局限 | 替代方案 | 需外部资源 |
|------|---------|---------|-----------|
| S4 | 统计代理误报高(18.2%) | Pinocchio + URDF FK 校验 | 机器人 URDF 文件 |
| C1 | 统计代理无法理解语义 | 三级 VLM 流水线 | VLM 模型(GPT-4V/Qwen-VL) |
| C2 | 无法验证视觉一致性 | URDF 渲染 + SAM3 IoU | 渲染引擎 + SAM3 |
| C3 | 无法检测像素缺陷 | 黑帧/模糊/损坏检测 | OpenCV 图像分析 |
