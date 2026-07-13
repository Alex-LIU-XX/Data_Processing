#!/usr/bin/env python3
"""可视化滤波前后的 state/action 轨迹对比，突显修改过的帧。

分别从原始 LeRobot 数据集和清洗后 parquet 加载数据，
对指定 episode 绘制每维度的 state/action 滤波前后对比 + 差异高亮。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGSIZE = (16, 2.6)


def _load_data_parquets(data_root: str) -> pd.DataFrame:
    data_dir = Path(data_root) / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def load_original(data_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _load_data_parquets(data_root)
    states = np.array(df["observation.state"].tolist())
    actions = np.array(df["action"].tolist())
    return states, actions, df["episode_index"].values


def load_filtered(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    states = np.array(df["observation.state"].tolist())
    actions = np.array(df["action"].tolist())
    return states, actions, df["episode_index"].values


def main():
    parser = argparse.ArgumentParser(description="Compare trajectories before/after filtering")
    parser.add_argument("--original", type=str,
                        default="example_data/pick_banana_100_newTable_1_offset_state",
                        help="Original dataset root")
    parser.add_argument("--filtered", type=str,
                        default="example_data/pick_banana_100_newTable_1_offset_state_cleaned/data/chunk-000/file-000.parquet",
                        help="Filtered parquet path")
    parser.add_argument("--episode", "-e", type=int, nargs="+", default=[0],
                        help="Episode indices to visualize")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: tests/results/filter_comparison)")
    args = parser.parse_args()

    out_dir = Path(args.out or Path(__file__).resolve().parent / "tests" / "results" / "filter_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    dim_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Gripper"]

    print("Loading original data...")
    st_orig, ac_orig, eps_orig = load_original(args.original)
    print(f"  Original: {len(st_orig)} frames")

    print("Loading filtered data...")
    st_filt, ac_filt, eps_filt = load_filtered(args.filtered)
    print(f"  Filtered: {len(st_filt)} frames")

    for ep in args.episode:
        mask_o = eps_orig == ep
        mask_f = eps_filt == ep
        s_o = st_orig[mask_o]
        s_f = st_filt[mask_f]
        a_o = ac_orig[mask_o]
        a_f = ac_filt[mask_f]

        T = len(s_o)
        D = s_o.shape[1]
        print(f"\nEpisode {ep}: {T} frames")

        # ─── 图1: State 对比 + 差异热力条 ────────────────────────────────
        fig, axes = plt.subplots(D + 1, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * (D + 1)),
                                 sharex=True, gridspec_kw={"height_ratios": [1] * D + [0.5]})

        for d in range(D):
            ax = axes[d]
            changed = ~np.isclose(s_o[:, d], s_f[:, d])
            changed_idx = np.where(changed)[0]

            # 差异填充（original 与 filtered 之间的区域）
            ax.fill_between(range(T), s_o[:, d], s_f[:, d],
                            where=changed, color="red", alpha=0.25, label="diff")

            # 轨迹线
            ax.plot(range(T), s_o[:, d], color="gray", linewidth=0.8, alpha=0.5, label="original")
            ax.plot(range(T), s_f[:, d], color="#1f77b4", linewidth=1.8, label="filtered")

            # 红点标记被修改的帧
            if len(changed_idx):
                ax.scatter(changed_idx, s_f[changed, d],
                           color="red", s=18, zorder=5, edgecolors="white", linewidth=0.4)

            ax.set_ylabel(dim_names[d])
            ax.grid(True, alpha=0.25)
            if d == 0:
                ax.legend(fontsize=7, ncol=3, loc="upper right")

        # 底部: 差异热力条（各维度修改帧的叠加指示）
        ax = axes[-1]
        combined = np.zeros(T, dtype=bool)
        for d in range(D):
            combined |= ~np.isclose(s_o[:, d], s_f[:, d])
        ax.imshow(combined[np.newaxis, :], aspect="auto", cmap="Reds",
                  extent=[0, T, 0, 1], alpha=0.8)
        ax.set_yticks([])
        ax.set_xlabel("Frame index")
        ax.set_title("Modified frames (any dim)", fontsize=9)

        fig.suptitle(f"Episode {ep} — State: original vs filtered  (red fill/highlight = modified)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = out_dir / f"ep{ep}_state_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)

        # ─── 图2: Action 对比 ─────────────────────────────────────────────
        fig, axes = plt.subplots(D, 1, figsize=(FIGSIZE[0], FIGSIZE[1] * D),
                                 sharex=True)

        for d in range(D):
            ax = axes[d]
            changed = ~np.isclose(a_o[:, d], a_f[:, d])
            changed_idx = np.where(changed)[0]

            ax.fill_between(range(T), a_o[:, d], a_f[:, d],
                            where=changed, color="red", alpha=0.25, label="diff")
            ax.plot(range(T), a_o[:, d], color="gray", linewidth=0.8, alpha=0.5, label="original")
            ax.plot(range(T), a_f[:, d], color="#d62728", linewidth=1.8, label="filtered")
            if len(changed_idx):
                ax.scatter(changed_idx, a_f[changed, d],
                           color="red", s=18, zorder=5, edgecolors="white", linewidth=0.4)
            ax.set_ylabel(dim_names[d])
            ax.grid(True, alpha=0.25)
            if d == 0:
                ax.legend(fontsize=7, ncol=3, loc="upper right")

        axes[-1].set_xlabel("Frame index")
        fig.suptitle(f"Episode {ep} — Action: original vs filtered  (red fill/highlight = modified)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = out_dir / f"ep{ep}_action_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)

        # ─── 图3: 四栏 —— state/action 各两栏 ──────────────────────────
        fig, axes = plt.subplots(D, 4, figsize=(28, 2.4 * D))
        for d in range(D):
            # col 0: state 对比
            ax = axes[d, 0]
            chg = ~np.isclose(s_o[:, d], s_f[:, d])
            ax.fill_between(range(T), s_o[:, d], s_f[:, d],
                            where=chg, color="red", alpha=0.2)
            ax.plot(range(T), s_o[:, d], color="gray", linewidth=0.6, alpha=0.4)
            ax.plot(range(T), s_f[:, d], color="#1f77b4", linewidth=1.5)
            if chg.any():
                ax.scatter(np.where(chg)[0], s_f[chg, d],
                           color="red", s=12, edgecolors="white", linewidth=0.3)
            ax.set_ylabel(dim_names[d])
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.set_title("State (filtered)", fontsize=10)

            # col 1: state 差异绝对值
            ax = axes[d, 1]
            delta = np.abs(s_o[:, d] - s_f[:, d])
            ax.fill_between(range(T), 0, delta, color="#1f77b4", alpha=0.5)
            ax.plot(range(T), delta, color="#1f77b4", linewidth=0.8)
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.set_title("|State delta|", fontsize=10)
            ax.set_ylim(bottom=0)

            # col 2: action 对比
            ax = axes[d, 2]
            chg = ~np.isclose(a_o[:, d], a_f[:, d])
            ax.fill_between(range(T), a_o[:, d], a_f[:, d],
                            where=chg, color="red", alpha=0.2)
            ax.plot(range(T), a_o[:, d], color="gray", linewidth=0.6, alpha=0.4)
            ax.plot(range(T), a_f[:, d], color="#d62728", linewidth=1.5)
            if chg.any():
                ax.scatter(np.where(chg)[0], a_f[chg, d],
                           color="red", s=12, edgecolors="white", linewidth=0.3)
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.set_title("Action (filtered)", fontsize=10)

            # col 3: action 差异绝对值
            ax = axes[d, 3]
            delta = np.abs(a_o[:, d] - a_f[:, d])
            ax.fill_between(range(T), 0, delta, color="#d62728", alpha=0.5)
            ax.plot(range(T), delta, color="#d62728", linewidth=0.8)
            ax.grid(True, alpha=0.2)
            if d == 0:
                ax.set_title("|Action delta|", fontsize=10)
            ax.set_ylim(bottom=0)

        axes[-1, 0].set_xlabel("Frame index")
        axes[-1, 1].set_xlabel("Frame index")
        axes[-1, 2].set_xlabel("Frame index")
        axes[-1, 3].set_xlabel("Frame index")
        fig.suptitle(f"Episode {ep} — State & Action: filtered overlay + delta")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = out_dir / f"ep{ep}_delta.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
