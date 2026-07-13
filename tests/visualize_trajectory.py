#!/usr/bin/env python3
"""Visualize state (blue) and action (red) trajectories: raw + normalized (q01→[-1,1]) per episode, per-dim histograms, multi-episode overlays and histograms.
python tests/visualize_trajectory.py --dataset-path example_data/pick_banana_100_newTable_1_offset_state --episode 0 1 2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.lerobot_dataset_adapter import LeRobotDatasetAdapter


def load_quantile_stats(stats_path: Path):
    with open(stats_path) as f:
        stats = json.load(f)
    state_q01 = np.array(stats["observation.state"]["q01"])
    state_q99 = np.array(stats["observation.state"]["q99"])
    action_q01 = np.array(stats["action"]["q01"])
    action_q99 = np.array(stats["action"]["q99"])
    state_mean = np.array(stats["observation.state"]["mean"])
    state_std = np.array(stats["observation.state"]["std"])
    action_mean = np.array(stats["action"]["mean"])
    action_std = np.array(stats["action"]["std"])
    return state_q01, state_q99, action_q01, action_q99, state_mean, state_std, action_mean, action_std


def normalize_q01(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    x = np.clip(x, q01, q99)
    return 2.0 * (x - q01) / (np.maximum(q99 - q01, 1e-8)) - 1.0


def normalize_zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def _central_diff(x: np.ndarray, fps: float, window_sec: float = 0.5) -> np.ndarray:
    """Central difference with smoothing over window_sec seconds."""
    w = max(1, int(round(fps * window_sec)))
    scale = fps / (2.0 * w)
    return (x[2 * w:] - x[:-2 * w]) * scale


def compute_kinematics(data: np.ndarray, fps: float, window_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vel = _central_diff(data, fps, window_sec)
    acc = _central_diff(vel, fps, window_sec)
    jerk = _central_diff(acc, fps, window_sec)
    return vel, acc, jerk


def load_episode(dataset, ep: int) -> tuple[np.ndarray, np.ndarray]:
    fr = dataset.episode_data_index["from"][ep].item()
    to = dataset.episode_data_index["to"][ep].item()
    states, actions = [], []
    for i in range(fr, to):
        frame = dataset[i]
        states.append(frame["observation.state"].numpy())
        actions.append(frame["action"].numpy())
    return np.array(states), np.array(actions)


def plot_per_episode_figure(states_raw, actions_raw, states_norm, actions_norm, ep, dataset_name, out_dir, dim_names, fps):
    dim = states_raw.shape[1]
    n = len(states_raw)

    fig, axes = plt.subplots(dim + 1, 3, figsize=(20, 2.5 * (dim + 1)))

    for i in range(dim):
        # column 0: raw trajectory
        ax = axes[i][0]
        ax.plot(range(n), states_raw[:, i], color="blue", linewidth=0.9, label="state")
        ax.plot(range(n), actions_raw[:, i], color="red", linewidth=0.9, label="action")
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Raw trajectory")
            ax.legend(loc="upper right", fontsize=7)

        # column 1: normalized trajectory
        ax = axes[i][1]
        ax.plot(range(n), states_norm[:, i], color="blue", linewidth=0.9, label="state")
        ax.plot(range(n), actions_norm[:, i], color="red", linewidth=0.9, label="action")
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        if i == 0:
            ax.set_title("Normalized [q01→[-1,1]]")
            ax.legend(loc="upper right", fontsize=7)

        # column 2: per-dim histogram (raw + normalized overlaid)
        ax = axes[i][2]
        ax.hist(states_raw[:, i], bins=80, color="blue", alpha=0.35, label="state raw")
        ax.hist(actions_raw[:, i], bins=80, color="red", alpha=0.35, label="action raw")
        ax.hist(states_norm[:, i], bins=80, color="blue", alpha=0.25, histtype="step", linewidth=1.5, label="state norm")
        ax.hist(actions_norm[:, i], bins=80, color="red", alpha=0.25, histtype="step", linewidth=1.5, label="action norm")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Per-dim histogram")
            ax.legend(loc="upper right", fontsize=7)

    # bottom row: combined histograms
    axes[-1][0].remove()
    ax_hist_raw = fig.add_subplot(dim + 1, 3, (3 * dim + 1))
    axes[-1][1].remove()
    ax_hist_norm = fig.add_subplot(dim + 1, 3, (3 * dim + 2))
    axes[-1][2].remove()
    ax_hist_combined = fig.add_subplot(dim + 1, 3, (3 * dim + 3))

    for data, color, label in [
        (states_raw, "blue", "state"), (actions_raw, "red", "action"),
    ]:
        ax_hist_raw.hist(data.flatten(), bins=100, color=color, alpha=0.4, label=label)
    ax_hist_raw.set_title("Raw histogram (all dims)")
    ax_hist_raw.set_xlabel("Value")
    ax_hist_raw.set_ylabel("Count")
    ax_hist_raw.legend(fontsize=7)

    for data, color, label in [
        (states_norm, "blue", "state"), (actions_norm, "red", "action"),
    ]:
        ax_hist_norm.hist(data.flatten(), bins=100, color=color, alpha=0.4, label=label)
    ax_hist_norm.set_title("Normalized histogram (all dims)")
    ax_hist_norm.set_xlabel("Value")
    ax_hist_norm.set_ylabel("Count")
    ax_hist_norm.legend(fontsize=7)

    # combined histogram: overlay raw + normalized
    for data, color, label in [
        (states_raw, "blue", "state raw"), (actions_raw, "red", "action raw"),
        (states_norm, "cyan", "state norm"), (actions_norm, "orange", "action norm"),
    ]:
        ax_hist_combined.hist(data.flatten(), bins=100, color=color, alpha=0.3, label=label)
    ax_hist_combined.set_title("Combined histogram")
    ax_hist_combined.set_xlabel("Value")
    ax_hist_combined.set_ylabel("Count")
    ax_hist_combined.legend(fontsize=7)

    axes[-1][0].set_xlabel("Frame index")
    fig.suptitle(f"Episode {ep} — {dataset_name}  [FPS: {fps}Hz] (state=blue, action=red)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"episode_{ep}_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_per_dim_histograms(states_raw, actions_raw, states_norm, actions_norm, ep, dataset_name, out_dir, dim_names, fps):
    """Separate figure: per-dim histograms showing raw + normalized distributions."""
    dim = states_raw.shape[1]
    fig, axes = plt.subplots(dim, 2, figsize=(14, 2.5 * dim))

    for i in range(dim):
        # left: raw histograms
        ax = axes[i][0]
        ax.hist(states_raw[:, i], bins=80, color="blue", alpha=0.5, label="state")
        ax.hist(actions_raw[:, i], bins=80, color="red", alpha=0.5, label="action")
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Raw histograms")
            ax.legend(fontsize=7)

        # right: normalized histograms
        ax = axes[i][1]
        ax.hist(states_norm[:, i], bins=80, color="blue", alpha=0.5, label="state")
        ax.hist(actions_norm[:, i], bins=80, color="red", alpha=0.5, label="action")
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Normalized histograms [q01→[-1,1]]")
            ax.legend(fontsize=7)

    axes[-1][0].set_xlabel("Value")
    axes[-1][1].set_xlabel("Value")
    fig.suptitle(f"Episode {ep} — Per-dim histograms  [FPS: {fps}Hz] (state=blue, action=red)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"episode_{ep}_per_dim_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_gaussian_histogram_figure(states_z, actions_z, ep, dataset_name, out_dir, dim_names, fps):
    """Per-dim histograms of z-score normalized data + combined histogram to assess Gaussian shape."""
    dim = states_z.shape[1]
    fig, axes = plt.subplots(dim + 1, 1, figsize=(10, 2.5 * (dim + 1)))

    for i in range(dim):
        ax = axes[i]
        ax.hist(states_z[:, i], bins=80, color="blue", alpha=0.5, label="state", density=True)
        ax.hist(actions_z[:, i], bins=80, color="red", alpha=0.5, label="action", density=True)
        # Overlay N(0,1) reference curve
        xx = np.linspace(-4, 4, 200)
        ax.plot(xx, np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi), color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
        ax.set_ylabel(dim_names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Gaussian (z-score) normalized — per dim", fontsize=11)
            ax.legend(fontsize=7)

    ax = axes[-1]
    ax.hist(states_z.flatten(), bins=100, color="blue", alpha=0.4, label="state", density=True)
    ax.hist(actions_z.flatten(), bins=100, color="red", alpha=0.4, label="action", density=True)
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi), color="black", linestyle="--", linewidth=1.2, label="N(0,1)")
    ax.set_title("Combined z-score histogram")
    ax.set_xlabel("z-score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Episode {ep} — Z-score normalized (Gaussian) — {dataset_name}  [FPS: {fps}Hz]")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"episode_{ep}_gaussian_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_multi_overlay(all_data, ep_list, dataset_name, out_dir, dim_names, suffix, title_suffix, fps, ylim=None):
    dim = all_data[0][0].shape[1]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 1, 1, 1, 1, 1))]

    fig, axes = plt.subplots(dim, 1, figsize=(14, 2.2 * dim), sharex=True)

    for idx, ep in enumerate(ep_list):
        ls = line_styles[idx % len(line_styles)]
        alpha = max(0.4, 1.0 - 0.15 * (idx // len(line_styles)))
        states, actions = all_data[idx]
        n = len(states)
        for i in range(dim):
            ax = axes[i]
            ax.plot(range(n), states[:, i], color="blue", linestyle=ls, alpha=alpha, linewidth=0.9,
                    label=f"state ep{ep}" if i == 0 else None)
            ax.plot(range(n), actions[:, i], color="red", linestyle=ls, alpha=alpha, linewidth=0.9,
                    label=f"action ep{ep}" if i == 0 else None)
            ax.set_ylabel(dim_names[i])
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(ylim)

    axes[0].legend(loc="upper right", fontsize=7, ncol=min(3, len(ep_list)))
    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"{title_suffix} — {dataset_name}  [FPS: {fps}Hz] (state=blue, action=red)")
    fig.tight_layout()

    out_path = out_dir / f"multi_episode_{suffix}_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_multi_histograms(all_data, ep_list, dataset_name, out_dir, suffix, title_suffix, fps):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, data_idx in [(axes[0], "state", 0), (axes[1], "action", 1)]:
        for idx, ep in enumerate(ep_list):
            data = all_data[idx][data_idx]
            alpha = max(0.3, 1.0 - 0.12 * idx)
            ax.hist(data.flatten(), bins=100, alpha=alpha, label=f"ep{ep}")
        ax.set_title(f"{title_suffix} {label}  [FPS: {fps}Hz]")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    out_path = out_dir / f"multi_episode_{suffix}_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_kinematics_figure(states, actions, fps, ep, dataset_name, out_dir, dim_names, norm=False):
    """Per-episode figure: 3 columns (velocity | acceleration | jerk) per dim + combined histograms."""
    dim = states.shape[1]
    s_vel, s_acc, s_jerk = compute_kinematics(states, fps)
    a_vel, a_acc, a_jerk = compute_kinematics(actions, fps)

    fig, axes = plt.subplots(dim + 1, 3, figsize=(20, 2.5 * (dim + 1)))
    derivs = [
        (s_vel, a_vel, "Velocity"),
        (s_acc, a_acc, "Acceleration"),
        (s_jerk, a_jerk, "Jerk"),
    ]

    for col, (s_d, a_d, title) in enumerate(derivs):
        for i in range(dim):
            ax = axes[i][col]
            n = len(s_d)
            ax.plot(range(n), s_d[:, i], color="blue", linewidth=0.9, label="state")
            ax.plot(range(n), a_d[:, i], color="red", linewidth=0.9, label="action")
            ax.set_ylabel(dim_names[i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(title if not norm else title + " (norm)")
                ax.legend(loc="upper right", fontsize=7)

    # bottom row: combined histograms
    for col in range(3):
        axes[-1][col].remove()
    ax_hists = [
        fig.add_subplot(dim + 1, 3, (3 * dim + 1)),
        fig.add_subplot(dim + 1, 3, (3 * dim + 2)),
        fig.add_subplot(dim + 1, 3, (3 * dim + 3)),
    ]
    for (s_d, a_d, title), ax in zip(derivs, ax_hists):
        ax.hist(s_d.flatten(), bins=100, color="blue", alpha=0.4, label="state")
        ax.hist(a_d.flatten(), bins=100, color="red", alpha=0.4, label="action")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    axes[-1][0].set_xlabel("Frame index")
    tag = "Normalized " if norm else ""
    fig.suptitle(f"Episode {ep} — {tag}Kinematics (0.5s window, FPS: {fps}Hz)  (state=blue, action=red)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    suffix = "_normalized" if norm else ""
    out_path = out_dir / f"episode_{ep}_kinematics{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_velocity_figure(state_vel, action_vel, state_vel_norm, action_vel_norm,
                          ep, fps, out_dir, dim_names):
    """7 × 4 subplot: vel raw trajectory | vel norm trajectory | hist raw | hist norm.

    Raw velocity  = diff(raw state/action)
    Norm velocity = diff(normalized state/action)
    """
    dim = state_vel.shape[1]
    n = len(state_vel)
    fig, axes = plt.subplots(dim, 4, figsize=(24, 2.5 * dim + 0.5))

    for d in range(dim):
        # 0: raw trajectory
        ax = axes[d, 0]
        ax.plot(range(n), state_vel[:, d], color="blue", linewidth=0.9, label="state")
        ax.plot(range(n), action_vel[:, d], color="red", linewidth=0.9, label="action")
        ax.set_ylabel(dim_names[d], fontsize=9)
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Velocity raw", fontsize=11)
            ax.legend(loc="upper right", fontsize=7)

        # 1: normalized trajectory
        ax = axes[d, 1]
        ax.plot(range(n), state_vel_norm[:, d], color="blue", linewidth=0.9, label="state")
        ax.plot(range(n), action_vel_norm[:, d], color="red", linewidth=0.9, label="action")
        ax.set_ylabel(dim_names[d], fontsize=9)
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Velocity norm (q01→[-1,1])", fontsize=11)
            ax.legend(loc="upper right", fontsize=7)

        # 2: histogram raw
        ax = axes[d, 2]
        ax.hist(state_vel[:, d], bins=60, color="blue", alpha=0.5, label="state")
        ax.hist(action_vel[:, d], bins=60, color="red", alpha=0.5, label="action")
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Histogram raw", fontsize=11)
            ax.legend(fontsize=7)

        # 3: histogram norm
        ax = axes[d, 3]
        ax.hist(state_vel_norm[:, d], bins=60, color="blue", alpha=0.5, label="state")
        ax.hist(action_vel_norm[:, d], bins=60, color="red", alpha=0.5, label="action")
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Histogram norm", fontsize=11)
            ax.legend(fontsize=7)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    axes[-1, 2].set_xlabel("Value")
    axes[-1, 3].set_xlabel("Value")
    fig.suptitle(f"Episode {ep} — Velocity (raw vs norm, FPS: {fps}Hz)  state=blue, action=red",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"episode_{ep}_velocity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_velocity_selfnorm_figure(state_vel, action_vel, ep, fps, out_dir, dim_names):
    """7 × 2 subplot: velocity self-normalized trajectory (left) + histogram (right).

    Computes per-dim q01/q99 from the combined velocity data itself, then normalizes
    to [-1, 1] using those stats (diff → normalize, unlike the existing "norm" column
    which normalizes → diff).
    """
    combined = np.concatenate([state_vel, action_vel], axis=0)
    v_q01 = np.percentile(combined, 1, axis=0)
    v_q99 = np.percentile(combined, 99, axis=0)
    s_vn = normalize_q01(state_vel, v_q01, v_q99)
    a_vn = normalize_q01(action_vel, v_q01, v_q99)

    dim = state_vel.shape[1]
    n = len(state_vel)
    fig, axes = plt.subplots(dim, 2, figsize=(16, 2.5 * dim + 0.5))

    for d in range(dim):
        ax = axes[d, 0]
        ax.plot(range(n), s_vn[:, d], color="blue", linewidth=0.9, label="state")
        ax.plot(range(n), a_vn[:, d], color="red", linewidth=0.9, label="action")
        ax.set_ylabel(dim_names[d], fontsize=9)
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Vel self-norm trajectory", fontsize=11)
            ax.legend(loc="upper right", fontsize=7)

        ax = axes[d, 1]
        ax.hist(s_vn[:, d], bins=60, color="blue", alpha=0.5, label="state")
        ax.hist(a_vn[:, d], bins=60, color="red", alpha=0.5, label="action")
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.set_title("Vel self-norm histogram", fontsize=11)
            ax.legend(fontsize=7)

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Value")
    fig.suptitle(f"Episode {ep} — Velocity self-normalized (q01/q99 from vel data, FPS: {fps}Hz)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = out_dir / f"episode_{ep}_velocity_selfnorm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_multi_kinematics(all_kin, ep_list, dataset_name, out_dir, dim_names, deriv_name, suffix, fps, norm=False):
    """Multi-episode overlay for a single derivative."""
    dim = all_kin[0][0].shape[1]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (3, 1, 1, 1, 1, 1))]

    fig, axes = plt.subplots(dim, 1, figsize=(14, 2.2 * dim), sharex=True)

    for idx, ep in enumerate(ep_list):
        ls = line_styles[idx % len(line_styles)]
        alpha = max(0.4, 1.0 - 0.15 * (idx // len(line_styles)))
        s_d, a_d = all_kin[idx]
        n = len(s_d)
        for i in range(dim):
            ax = axes[i]
            ax.plot(range(n), s_d[:, i], color="blue", linestyle=ls, alpha=alpha, linewidth=0.9,
                    label=f"state ep{ep}" if i == 0 else None)
            ax.plot(range(n), a_d[:, i], color="red", linestyle=ls, alpha=alpha, linewidth=0.9,
                    label=f"action ep{ep}" if i == 0 else None)
            ax.set_ylabel(dim_names[i])
            ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=7, ncol=min(3, len(ep_list)))
    axes[-1].set_xlabel("Frame index")
    tag = " (norm)" if norm else ""
    fig.suptitle(f"{deriv_name} (0.5s window, FPS: {fps}Hz){tag} — {dataset_name}  (state=blue, action=red)")
    fig.tight_layout()

    suffix_tag = f"{suffix}_normalized" if norm else suffix
    out_path = out_dir / f"multi_episode_{suffix_tag}_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw + normalized state/action trajectories, per-episode and multi-episode overlays + histograms."
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--episode", "-e", type=int, nargs="+", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    repo_id = f"{dataset_path.parent.name}/{dataset_path.name}"
    episodes = args.episode
    dataset_name = dataset_path.name

    print(f"Loading dataset: {repo_id} from {dataset_path}")
    dataset = LeRobotDatasetAdapter(repo_id, root=str(dataset_path), delta_timestamps=None)
    fps = float(dataset.meta.fps)

    stats_path = dataset_path / "meta" / "stats.json"
    state_q01, state_q99, action_q01, action_q99, state_mean, state_std, action_mean, action_std = load_quantile_stats(stats_path)
    print(f"Using quantile stats from {stats_path}, fps={fps}")

    out_dir = Path(__file__).resolve().parent / "results" / "visualize_trajectory" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dim_names = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Gripper"]

    all_raw = []
    all_norm = []
    all_vel = []
    all_acc = []
    all_jerk = []
    all_vel_norm = []
    all_acc_norm = []
    all_jerk_norm = []
    for ep in episodes:
        states, actions = load_episode(dataset, ep)
        states_n = normalize_q01(states, state_q01, state_q99)
        actions_n = normalize_q01(actions, action_q01, action_q99)
        states_z = normalize_zscore(states, state_mean, state_std)
        actions_z = normalize_zscore(actions, action_mean, action_std)
        all_raw.append((states, actions))
        all_norm.append((states_n, actions_n))
        print(f"  Episode {ep}: {len(states)} frames")

        # per-episode trajectory figure (3 columns)
        plot_per_episode_figure(states, actions, states_n, actions_n, ep, dataset_name, out_dir, dim_names, fps)

        # per-dim histogram figure
        plot_per_dim_histograms(states, actions, states_n, actions_n, ep, dataset_name, out_dir, dim_names, fps)

        # kinematics (raw)
        s_vel, s_acc, s_jerk = compute_kinematics(states, fps)
        a_vel, a_acc, a_jerk = compute_kinematics(actions, fps)
        all_vel.append((s_vel, a_vel))
        all_acc.append((s_acc, a_acc))
        all_jerk.append((s_jerk, a_jerk))
        plot_kinematics_figure(states, actions, fps, ep, dataset_name, out_dir, dim_names)

        # kinematics (normalized)
        sn_vel, sn_acc, sn_jerk = compute_kinematics(states_n, fps)
        an_vel, an_acc, an_jerk = compute_kinematics(actions_n, fps)
        all_vel_norm.append((sn_vel, an_vel))
        all_acc_norm.append((sn_acc, an_acc))
        all_jerk_norm.append((sn_jerk, an_jerk))
        plot_kinematics_figure(states_n, actions_n, fps, ep, dataset_name, out_dir, dim_names, norm=True)

        # velocity comparison: raw vs normalized
        plot_velocity_figure(s_vel, a_vel, sn_vel, an_vel, ep, fps, out_dir, dim_names)

        # velocity self-normalized: diff → normalize
        plot_velocity_selfnorm_figure(s_vel, a_vel, ep, fps, out_dir, dim_names)

        # z-score normalized histogram (Gaussian)
        plot_gaussian_histogram_figure(states_z, actions_z, ep, dataset_name, out_dir, dim_names, fps)

    # multi-episode overlays
    plot_multi_overlay(all_raw, episodes, dataset_name, out_dir, dim_names, "raw", "Raw trajectories", fps)
    plot_multi_overlay(all_norm, episodes, dataset_name, out_dir, dim_names, "normalized",
                       "Normalized trajectories [q01→[-1,1]]", fps, ylim=(-1.5, 1.5))

    # multi-episode histograms
    plot_multi_histograms(all_raw, episodes, dataset_name, out_dir, "raw", "Raw", fps)
    plot_multi_histograms(all_norm, episodes, dataset_name, out_dir, "normalized", "Normalized", fps)

    # multi-episode kinematics overlays (raw)
    for all_k, suffix, name in [(all_vel, "vel", "Velocity"), (all_acc, "acc", "Acceleration"), (all_jerk, "jerk", "Jerk")]:
        plot_multi_kinematics(all_k, episodes, dataset_name, out_dir, dim_names, name, suffix, fps)

    # multi-episode kinematics overlays (normalized)
    for all_k, suffix, name in [(all_vel_norm, "vel", "Velocity"), (all_acc_norm, "acc", "Acceleration"), (all_jerk_norm, "jerk", "Jerk")]:
        plot_multi_kinematics(all_k, episodes, dataset_name, out_dir, dim_names, name, suffix, fps, norm=True)

    print("Done.")


if __name__ == "__main__":
    main()
