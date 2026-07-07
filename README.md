# Data Processing — LeRobot Dataset Utilities

A collection of data loading, visualization, and quality-analysis utilities for
LeRobot-format robot demonstration datasets (v3.0).

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [1. Dataset Adapter (`data/`)](#1-dataset-adapter-data)
  - [2. Signal Analysis — Stage 1 (`utils/signal_analysis.py`)](#2-signal-analysis--stage-1)
  - [3. Trend Alignment — Stage 2 (`utils/trend_alignment.py`)](#3-trend-alignment--stage-2)
  - [4. Trajectory Visualization (`tests/`)](#4-trajectory-visualization)
- [How to Run Tests](#how-to-run-tests)
- [Output Structure](#output-structure)

---

## Setup

### Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) package manager

### Create Environment and Install Dependencies

```bash
cd Data_Processing/

# Create virtual environment
uv venv --python 3.10

# Install all dependencies
unset VIRTUAL_ENV && uv sync
```

This creates `.venv/` and installs all dependencies from `pyproject.toml`,
including `lerobot` (pinned git revision), `torch`, `scipy`, `matplotlib`,
`datasets`, `pandas`, `numpy`, etc.

### Activate the Environment

```bash
source .venv/bin/activate
```

All subsequent `python` commands use the project venv.

---

## Project Structure

```
/home/lxx/repo/Data_Processing/
├── pyproject.toml                  # Dependencies + project config
├── README.md                       # This file
├── data/                           # Data loading utilities
│   ├── __init__.py
│   └── lerobot_dataset_adapter.py  # LeRobotDataset + Metadata adapter (v3.0)
├── utils/                          # Analysis utilities
│   ├── __init__.py
│   ├── signal_analysis.py          # Stage 1: smoothing, deviation, sudden changes
│   └── trend_alignment.py          # Stage 2: lag detection, directional agreement
├── tests/                          # Test scripts + results
│   ├── conftest.py                 # pytest hooks for result logging
│   ├── test_lerobot_dataset_adapter.py
│   ├── test_signal_analysis.py
│   ├── test_trend_alignment.py
│   ├── visualize_trajectory.py     # Multi-figure trajectory plots
│   └── results/                    # Automatically generated output
│       ├── test_lerobot_dataset_adapter/
│       ├── test_signal_analysis/
│       ├── test_trend_alignment/
│       └── visualize_trajectory/
└── doc/                            # Documentation
    ├── dev/development_log.md       # Development history
    └── data_analysis_report.md
```

---

## Modules

### 1. Dataset Adapter (`data/`)

**File:** `data/lerobot_dataset_adapter.py`

Copied from the `openpi` project. Supports both v2.x (JSONL-based) and v3.0
(parquet-based) LeRobot dataset formats.

```python
from data.lerobot_dataset_adapter import (
    LeRobotDatasetAdapter,
    LeRobotDatasetMetadataAdapter,
    detect_dataset_version,
)
```

**Usage:**

```python
meta = LeRobotDatasetMetadataAdapter(
    "miku112/pick_banana_100_newTable_1_offset_state",
    root="/path/to/dataset",
)
print(f"Episodes: {meta.total_episodes}, FPS: {meta.fps}")

dataset = LeRobotDatasetAdapter(
    "miku112/pick_banana_100_newTable_1_offset_state",
    root="/path/to/dataset",
    delta_timestamps=None,
)
frame = dataset[0]  # dict with observation.state, observation.images.image, action, ...
```

---

### 2. Signal Analysis — Stage 1 (`utils/signal_analysis.py`)

Implements trajectory smoothing and multi-level sudden change detection.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `extract_trend(data, median_windows, sg_window, sg_order)` | Cascaded median + Savitzky-Golay smoothing |
| `compute_deviation_signals(data, smoothed, fps)` | Residual, acceleration (2nd diff), jerk (3rd diff) |
| `detect_sudden_changes(data, smoothed, fps, ThresholdConfig)` | Triple-threshold AND detection |
| `compute_adaptive_threshold(signal, method, k, percentile)` | Adaptive threshold (MAD / typical_scale / percentile) |

**ThresholdConfig:**

```python
from utils.signal_analysis import ThresholdConfig

cfg = ThresholdConfig(
    residual_method="mad",      # "mad", "typical_scale", or "percentile"
    residual_k=3.0,
    acceleration_method="mad",
    acceleration_k=3.0,
    jerk_method="mad",
    jerk_k=3.0,
    dim_wise=False,             # True = per-dim thresholds
)
```

**Usage:**

```python
from utils.signal_analysis import extract_trend, detect_sudden_changes

smoothed = extract_trend(actions)  # cascaded median + SG filter
result = detect_sudden_changes(actions, smoothed, fps=10.0)

events = result["events"]                    # (T,) bool array
num_events = events.sum()
print(f"Sudden changes: {num_events}")
```

---

### 3. Trend Alignment — Stage 2 (`utils/trend_alignment.py`)

Implements state-action temporal alignment validation via cross-correlation
and directional agreement checking.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `compute_optimal_lag(source, target, max_lag, method)` | Find lag that maximizes Pearson correlation |
| `directional_agreement(aligned_source, target, tolerance)` | Sign-consistency of first-order differences |
| `analyze_trend_alignment(actions, states, AlignmentConfig)` | End-to-end: lag → alignment → DA → flags |
| `align_signals(source, target, lag)` | Trim both arrays to overlapping region |

**Lag sign convention:**

| Lag | Meaning | Suspicious? |
|-----|---------|-------------|
| `< 0` | Action leads state | No (causal) |
| `= 0` | In sync | No |
| `> 0` | Action lags state | Yes (violates causality) |

**Usage:**

```python
from utils.trend_alignment import AlignmentConfig, analyze_trend_alignment

cfg = AlignmentConfig(max_lag=30)
result = analyze_trend_alignment(actions, states, cfg)

print(f"Optimal lag: {result['lag']['optimal_lag']} frames")
print(f"DA score:    {result['da']['da_overall']:.3f}")
print(f"Suspicious:  {result['is_suspicious']}")
```

---

### 4. Trajectory Visualization

**File:** `tests/visualize_trajectory.py`

Generates a comprehensive set of plots for raw and normalized trajectories,
including kinematics (velocity/acceleration/jerk), per-dim histograms, and
multi-episode overlays.

```bash
uv run python tests/visualize_trajectory.py \
    --dataset-path /path/to/dataset \
    --episode 0 1 2
```

---

## How to Run Tests

All test scripts save figures and JSON results to `tests/results/<test_name>/run_<timestamp>/`.

### Test Suite

```bash
# Run all tests (with uv)
unset VIRTUAL_ENV && uv run python -m pytest tests/ -v

# Run a specific test file
uv run python -m pytest tests/test_lerobot_dataset_adapter.py -v
```

### Individual Analysis Scripts

```bash
# Stage 1: Sudden change detection
uv run python tests/test_signal_analysis.py

# Stage 2: Trend alignment analysis
uv run python tests/test_trend_alignment.py

# Trajectory visualization
uv run python tests/visualize_trajectory.py \
    --dataset-path /home/lxx/repo/datasets/lerobot/miku112/pick_banana_100_newTable_1_offset_state \
    --episode 0 1
```

---

## Output Structure

All test results follow the same convention:

```
tests/results/<test_name>/run_<YYYYMMDD_HHMMSS>/
├── results.json    # Machine-readable metrics
├── ep0_*.png       # Per-episode figures
├── ep1_*.png
├── ...
└── summary.png     # Cross-episode summary
```

The `results.json` file contains the full set of metrics for downstream
processing and reporting.
