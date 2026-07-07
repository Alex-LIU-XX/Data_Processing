"""Test trend alignment (Stage 2) on real LeRobot dataset. Saves figures + JSON."""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.lerobot_dataset_adapter import LeRobotDatasetAdapter
from utils.trend_alignment import AlignmentConfig, analyze_trend_alignment

DATASET_ROOT = Path("/home/lxx/repo/datasets/lerobot/miku112/pick_banana_100_newTable_1_offset_state")
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


def plot_cross_correlation(lags, per_dim_corrs, agg_corr, optimal_lag, optimal_corr, out_path, dim_names):
    D = per_dim_corrs.shape[1]
    fig, axes = plt.subplots(D + 1, 1, figsize=(12, 2.2 * (D + 1)), sharex=True)
    if D + 1 == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]
        ax.plot(lags, per_dim_corrs[:, d], linewidth=1.0, alpha=0.8)
        opt_d_idx = np.nanargmax(per_dim_corrs[:, d])
        ax.axvline(x=lags[opt_d_idx], color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(dim_names[d] if d < len(dim_names) else f"Dim {d}")
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(lags, agg_corr, color="black", linewidth=1.8, label="aggregated")
    ax.axvline(x=optimal_lag, color="red", linestyle="--", linewidth=1.5,
               label=f"optimal lag={optimal_lag}  (r={optimal_corr:.3f})")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-correlation: Action vs State", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_optimal_lag_bars(per_dim_lags, per_dim_corrs, optimal_lag, out_path, dim_names):
    D = len(per_dim_lags)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(D)
    colors = ["green" if v == optimal_lag else "steelblue" for v in per_dim_lags]
    ax.bar(x, per_dim_lags, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(y=optimal_lag, color="red", linestyle="--", linewidth=1.2, label=f"overall optimal = {optimal_lag}")
    ax.set_xticks(x)
    ax.set_xticklabels([dim_names[i] if i < len(dim_names) else str(i) for i in range(D)], rotation=30)
    ax.set_ylabel("Optimal lag (frames)")
    ax.set_title("Optimal lag per dimension")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_alignment_overlay(aligned_action, aligned_state, action_raw_trim, state_raw_trim, lag, out_path, dim_names):
    T, D = aligned_action.shape
    fig, axes = plt.subplots(D, 1, figsize=(14, 2.0 * D), sharex=True)
    if D == 1:
        axes = [axes]
    for d in range(D):
        ax = axes[d]
        ax.plot(state_raw_trim[:, d], color="blue", alpha=0.4, linewidth=0.7, label="state (raw)")
        ax.plot(action_raw_trim[:, d], color="red", alpha=0.4, linewidth=0.7, label="action (raw)")
        ax.plot(aligned_state[:, d], color="blue", linewidth=1.5, label="state (aligned)")
        ax.plot(aligned_action[:, d], color="red", linewidth=1.5, label="action (aligned)")
        ax.set_ylabel(dim_names[d] if d < len(dim_names) else f"Dim {d}")
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"State vs Action alignment  (lag={lag} frames)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_da_heatmap(da_per_frame, out_path, dim_names):
    T_minus_1, D = da_per_frame.shape
    fig, ax = plt.subplots(figsize=(14, 0.6 * D + 2))
    im = ax.imshow(da_per_frame.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(D))
    ax.set_yticklabels([dim_names[d] if d < len(dim_names) else str(d) for d in range(D)])
    ax.set_xlabel("Frame index")
    ax.set_title("Directional Agreement per dimension (green=agree, red=disagree)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Agreement")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def plot_summary(results, out_dir):
    episodes = [r["episode"] for r in results]
    lags = [r["optimal_lag"] for r in results]
    corrs = [r["optimal_correlation"] for r in results]
    das = [r["da_overall"] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].bar(episodes, lags, color="steelblue", alpha=0.8)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Optimal lag (frames)")
    axes[0].set_title("Optimal lag per episode")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(episodes, corrs, color="steelblue", alpha=0.8)
    axes[1].axhline(y=0.5, color="orange", linestyle="--", linewidth=1.0, label="r=0.5")
    axes[1].set_ylabel("Max correlation")
    axes[1].set_title("Max cross-correlation per episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(episodes, das, color="green" if all(d >= 0.6 for d in das) else "red", alpha=0.8)
    axes[2].axhline(y=0.6, color="orange", linestyle="--", linewidth=1.0, label="DA=0.6 threshold")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("DA score")
    axes[2].set_title("Directional Agreement per episode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


def main():
    episodes = [0, 1, 2, 3, 4]

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
        / Path(__file__).stem
        / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATASET_ROOT}")
    dataset = LeRobotDatasetAdapter(
        "miku112/pick_banana_100_newTable_1_offset_state",
        root=str(DATASET_ROOT),
        delta_timestamps=None,
    )

    cfg = AlignmentConfig(max_lag=30)
    all_results = []

    for ep in episodes:
        print(f"\n=== Episode {ep} ===")
        states, actions = load_episode(dataset, ep)
        T = len(states)
        print(f"  Frames: {T}")

        result = analyze_trend_alignment(actions, states, cfg)
        lag = result["lag"]
        da = result["da"]

        print(f"  Optimal lag: {lag['optimal_lag']} frames (corr={lag['optimal_correlation']:.3f})")
        print(f"  DA overall:   {da['da_overall']:.3f}")
        print(f"  Disagreement frames: {len(da['disagreement_frames'])}")
        print(f"  Suspicious:   {result['is_suspicious']}")

        # --- Figures ---

        # Cross-correlation curve
        plot_cross_correlation(
            lag["lags"], lag["per_dim_correlations"], lag["correlations"],
            lag["optimal_lag"], lag["optimal_correlation"],
            results_dir / f"ep{ep}_cross_correlation.png", DIM_NAMES,
        )

        # Optimal lag per dim
        plot_optimal_lag_bars(
            lag["per_dim_optimal_lags"], lag["per_dim_optimal_corrs"],
            lag["optimal_lag"],
            results_dir / f"ep{ep}_lag_bars.png", DIM_NAMES,
        )

        # Alignment overlay
        aligned_action = result["aligned_action"]
        aligned_state = result["aligned_state"]
        shift = lag["optimal_lag"]
        if shift >= 0:
            action_raw_trim = actions[shift:]
            state_raw_trim = states[:T - shift]
        else:
            action_raw_trim = actions[:T + shift]
            state_raw_trim = states[-shift:]
        plot_alignment_overlay(
            aligned_action, aligned_state, action_raw_trim, state_raw_trim,
            shift, results_dir / f"ep{ep}_alignment_overlay.png", DIM_NAMES,
        )

        # DA heatmap
        plot_da_heatmap(
            da["da_per_frame"],
            results_dir / f"ep{ep}_da_heatmap.png", DIM_NAMES,
        )

        all_results.append({
            "episode": ep,
            "num_frames": T,
            "optimal_lag": lag["optimal_lag"],
            "optimal_correlation": round(lag["optimal_correlation"], 4),
            "per_dim_optimal_lags": [round(v, 1) for v in lag["per_dim_optimal_lags"]],
            "per_dim_optimal_corrs": [round(v, 4) for v in lag["per_dim_optimal_corrs"]],
            "da_overall": round(da["da_overall"], 4),
            "da_per_dim": [round(v, 4) for v in da["da_per_dim"]],
            "num_disagreement_frames": len(da["disagreement_frames"]),
            "disagreement_frames": da["disagreement_frames"],
            "is_suspicious": result["is_suspicious"],
            "is_suspicious_lag": result["is_suspicious_lag"],
            "is_suspicious_da": result["is_suspicious_da"],
        })

    # Summary figure
    plot_summary(all_results, results_dir)

    # JSON results
    output = {
        "dataset": str(DATASET_ROOT),
        "config": {
            "max_lag": cfg.max_lag,
            "correlation_method": cfg.correlation_method,
            "da_tolerance": cfg.da_tolerance,
            "suspicious_lag_ratio": cfg.suspicious_lag_ratio,
            "suspicious_da_threshold": cfg.suspicious_da_threshold,
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
