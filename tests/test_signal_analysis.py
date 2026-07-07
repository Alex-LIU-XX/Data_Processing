"""Test signal analysis utils on real LeRobot dataset. Saves figures + JSON results."""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.lerobot_dataset_adapter import LeRobotDatasetAdapter
from utils.signal_analysis import (
    ThresholdConfig,
    detect_sudden_changes,
    extract_trend,
)

DATASET_ROOT = Path("/home/lxx/repo/datasets/lerobot/miku112/pick_banana_100_newTable_1_offset_state")
RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / Path(__file__).stem
    / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

DIM_NAMES = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5", "Joint 6", "Gripper"]


def load_episode(dataset, ep: int) -> tuple[np.ndarray, np.ndarray]:
    fr = dataset.episode_data_index["from"][ep].item()
    to = dataset.episode_data_index["to"][ep].item()
    states, actions = [], []
    for i in range(fr, to):
        frame = dataset[i]
        states.append(frame["observation.state"].numpy())
        actions.append(frame["action"].numpy())
    return np.array(states), np.array(actions)


def plot_trajectory_with_events(data, smoothed, events, title, out_path, dim_names, fps):
    T, D = data.shape
    fig, axes = plt.subplots(D, 1, figsize=(16, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]
    for d in range(D):
        ax = axes[d]
        ax.plot(range(T), data[:, d], color="blue", alpha=0.5, linewidth=0.7, label="raw")
        ax.plot(range(T), smoothed[:, d], color="orange", linewidth=1.5, label="smoothed")
        event_frames = np.where(events)[0]
        if len(event_frames) > 0:
            ax.scatter(event_frames, data[event_frames, d], color="red", s=20, zorder=5, label="sudden change")
        ax.set_ylabel(dim_names[d] if d < len(dim_names) else f"Dim {d}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_deviation_signals(dev, thresh, title, out_path, fps):
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    labels = ["Residual", "Acceleration", "Jerk"]
    keys = ["residual_deviation", "acceleration_deviation", "jerk_deviation"]
    thresh_keys = ["residual_threshold", "acceleration_threshold", "jerk_threshold"]
    colors = ["green", "purple", "brown"]

    for ax, label, key, tk, color in zip(axes, labels, keys, thresh_keys, colors):
        sig = dev[key]
        ax.plot(sig, color=color, alpha=0.7, linewidth=0.8)
        t = dev[tk]
        if isinstance(t, list):
            t = t[0] if len(t) > 0 else 0
        ax.axhline(y=t, color="red", linestyle="--", linewidth=1.2, label=f"threshold={t:.3f}")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_summary_heatmap(all_events, out_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    n_eps = len(all_events)
    max_len = max(len(e) for e in all_events) if all_events else 1
    heatmap = np.full((n_eps, max_len), np.nan)
    for i, e in enumerate(all_events):
        heatmap[i, :len(e)] = e.astype(float)
    im = ax.imshow(heatmap, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Episode")
    ax.set_title("Sudden change events across episodes")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Event flag")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    episodes = [0, 1, 2, 3, 4]
    fps = 10.0

    print(f"Loading dataset from {DATASET_ROOT}")
    dataset = LeRobotDatasetAdapter(
        "miku112/pick_banana_100_newTable_1_offset_state",
        root=str(DATASET_ROOT),
        delta_timestamps=None,
    )

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    threshold_cfg = ThresholdConfig()

    all_results = []
    all_events_state = []
    all_events_action = []

    for ep in episodes:
        print(f"\n=== Episode {ep} ===")
        states, actions = load_episode(dataset, ep)
        T = len(states)
        print(f"  Frames: {T}")

        # --- State analysis ---
        state_smoothed = extract_trend(states)
        state_dev = detect_sudden_changes(states, state_smoothed, fps=fps, threshold_cfg=threshold_cfg)
        n_state_events = int(state_dev["events"].sum())
        pct_state = float(n_state_events / T * 100)
        print(f"  State sudden changes: {n_state_events}/{T} ({pct_state:.1f}%)")

        # --- Action analysis ---
        action_smoothed = extract_trend(actions)
        action_dev = detect_sudden_changes(actions, action_smoothed, fps=fps, threshold_cfg=threshold_cfg)
        n_action_events = int(action_dev["events"].sum())
        pct_action = float(n_action_events / T * 100)
        print(f"  Action sudden changes: {n_action_events}/{T} ({pct_action:.1f}%)")

        # --- Figures ---
        # Trajectory with events
        plot_trajectory_with_events(
            states, state_smoothed, state_dev["events"],
            f"Episode {ep} — State trajectory", results_dir / f"ep{ep}_state_trajectory.png",
            DIM_NAMES, fps,
        )
        plot_trajectory_with_events(
            actions, action_smoothed, action_dev["events"],
            f"Episode {ep} — Action trajectory", results_dir / f"ep{ep}_action_trajectory.png",
            DIM_NAMES, fps,
        )

        # Deviation signals
        plot_deviation_signals(
            state_dev, None, f"Episode {ep} — State deviation signals",
            results_dir / f"ep{ep}_state_deviations.png", fps,
        )
        plot_deviation_signals(
            action_dev, None, f"Episode {ep} — Action deviation signals",
            results_dir / f"ep{ep}_action_deviations.png", fps,
        )

        all_events_state.append(state_dev["events"])
        all_events_action.append(action_dev["events"])

        all_results.append({
            "episode": ep,
            "num_frames": T,
            "state": {
                "num_sudden_changes": n_state_events,
                "pct_frames_changed": round(pct_state, 2),
                "sudden_change_frames": np.where(state_dev["events"])[0].tolist(),
                "residual_threshold": state_dev["residual_threshold"],
                "acceleration_threshold": state_dev["acceleration_threshold"],
                "jerk_threshold": state_dev["jerk_threshold"],
            },
            "action": {
                "num_sudden_changes": n_action_events,
                "pct_frames_changed": round(pct_action, 2),
                "sudden_change_frames": np.where(action_dev["events"])[0].tolist(),
                "residual_threshold": action_dev["residual_threshold"],
                "acceleration_threshold": action_dev["acceleration_threshold"],
                "jerk_threshold": action_dev["jerk_threshold"],
            },
        })

    # Multi-episode summary heatmaps
    plot_summary_heatmap(all_events_state, results_dir / "summary_state_events.png")
    plot_summary_heatmap(all_events_action, results_dir / "summary_action_events.png")

    # Save JSON results
    output = {
        "dataset": str(DATASET_ROOT),
        "fps": fps,
        "threshold_config": {
            "residual_method": threshold_cfg.residual_method,
            "residual_k": threshold_cfg.residual_k,
            "acceleration_method": threshold_cfg.acceleration_method,
            "acceleration_k": threshold_cfg.acceleration_k,
            "jerk_method": threshold_cfg.jerk_method,
            "jerk_k": threshold_cfg.jerk_k,
            "dim_wise": threshold_cfg.dim_wise,
        },
        "results": all_results,
    }

    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
