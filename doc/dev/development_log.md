# Development Log — Data Processing Utilities

## Overview

This document records the development history of the `Data_Processing` project,
a collection of utilities for loading, analyzing, and validating LeRobot-format
robot demonstration datasets.

---

## 2026-07-07 — Initial Project Setup

### Files Created

| File | Description |
|------|-------------|
| `pyproject.toml` | Project configuration with uv-based dependency management |
| `data/__init__.py` | Package init for data loading utilities |
| `tests/conftest.py` | pytest hooks for saving test results as JSON |
| `.gitignore` | Standard Python gitignore |

### Dependencies

```
datasets>=3.0.0, lerobot (git), numpy, pandas, packaging, pyarrow,
scipy>=1.12.0, torch>=2.0.0
```

### Decisions

- Python 3.10+ target (uv venv).
- `lerobot` pinned to the same git revision as the upstream `openpi` project.
- Test results saved to `tests/results/<test_name>/run_<timestamp>/` for traceability.

---

## 2026-07-07 — LeRobot Dataset Adapter

### Task

Decouple `lerobot_dataset_adapter.py` from the `openpi` codebase and copy it to
this project as a standalone data loading utility.

### Files

| File | Description |
|------|-------------|
| `data/lerobot_dataset_adapter.py` | Copied from `openpi-Alex/src/openpi/training/` (zero openpi imports) |
| `tests/test_lerobot_dataset_adapter.py` | 50 tests adapted for `pick_banana_100_newTable_1_offset_state` dataset |

### Adaptations

- Import path changed from `openpi.training.lerobot_dataset_adapter` to
  `data.lerobot_dataset_adapter`.
- `torch.stack(self.hf_dataset["..."]).numpy()` → `torch.tensor(self.hf_dataset["..."]).numpy()`
  for compatibility with `datasets>=5.0`.

### Test Results

- **49/50 passed** (1 failure is upstream `lerobot` bug with `datasets>=5.0`).

---

## 2026-07-07 — Trajectory Visualization

### Task

Migrate `visualize_trajectory.py` from `openpi-Alex/tests/`.

### Files

| File | Description |
|------|-------------|
| `tests/visualize_trajectory.py` | Multi-figure trajectory analysis (raw, normalized, kinematics, z-score) |

### Adaptations

- Import path changed.
- Output path changed to `tests/results/visualize_trajectory/`.

---

## 2026-07-07 — Signal Analysis (Stage 1)

### Task

Implement "Stage 1" trajectory smoothing + sudden change detection.

### Methods

1. **Extract trend**: Cascaded median filtering (`scipy.signal.medfilt`) +
   Savitzky-Golay smoothing (`scipy.signal.savgol_filter`).
2. **Deviation signals**:
   - Residual: `|raw - smoothed|`
   - Acceleration: second-order central difference × fps²
   - Jerk: third-order central difference × fps³
3. **Sudden change detection**: A frame is flagged only when *all three*
   deviation signals exceed their adaptive thresholds simultaneously.

### Threshold Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| `mad` | `median + k × MAD` | Non-negative skewed signals (default) |
| `typical_scale` | `k × median(\|signal\|)` | Scale-invariant comparison |
| `percentile` | Given percentile | Fixed false-positive rate |

### Files

| File | Description |
|------|-------------|
| `utils/signal_analysis.py` | `extract_trend`, `compute_deviation_signals`, `detect_sudden_changes`, `ThresholdConfig` |
| `tests/test_signal_analysis.py` | 5-episode analysis on `pick_banana` dataset |

### Detection Results (pick_banana, MAD k=3)

~1% of frames flagged as sudden changes (smooth pick-and-place task).

---

## 2026-07-07 — Trend Alignment (Stage 2)

### Task

Implement "Stage 2" state-action temporal alignment validation.

### Methods

#### Cross-correlation Delay Detection

Shift action signal relative to state over `[-max_lag, +max_lag]` frames.
At each shift, compute per-dim Pearson correlation, then aggregate (mean/median).
The optimal lag maximizes the aggregated correlation.

**Lag sign convention:**

| Lag | Meaning | Verdict |
|-----|---------|---------|
| `lag < 0` | Action leads state by \|lag\| frames | OK |
| `lag = 0` | Action and state in sync | OK |
| `lag > 0` | Action lags state | Suspicious (violates causality) |

#### Directional Agreement (DA)

After aligning by optimal lag, compare first-order forward differences:

```
agree[i,d] = sign(action_diff[i,d]) == sign(state_diff[i,d])
```

DA overall = fraction of frames where directions agree.

A DA score < 0.6 flags the episode as suspicious.

### Files

| File | Description |
|------|-------------|
| `utils/trend_alignment.py` | `compute_optimal_lag`, `directional_agreement`, `analyze_trend_alignment`, `AlignmentConfig` |
| `tests/test_trend_alignment.py` | 5-episode analysis on `pick_banana` dataset |

### Alignment Results (pick_banana)

| Episode | Lag | Correlation | DA | Suspicious |
|---------|-----|-------------|----|------------|
| 0–4 | -1 (action leads by 1) | 1.000 | 1.000 | False |

Dataset is perfectly aligned (action[t] = state[t+1]).

---

## Known Issues

- **`test_with_delta_timestamps`** in `test_lerobot_dataset_adapter.py` fails
  due to upstream `lerobot` incompatibility with `datasets>=5.0`
  (`torch.stack` on `Column` object). Not a project bug.
