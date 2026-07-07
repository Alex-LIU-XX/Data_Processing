"""Trajectory signal analysis: smoothing, deviation signals, sudden change detection."""

import dataclasses
from typing import Literal

import numpy as np
from scipy.signal import medfilt, savgol_filter


@dataclasses.dataclass
class ThresholdConfig:
    """Configuration for adaptive threshold computation.

    Each deviation signal (residual, acceleration, jerk) has its own method and
    sensitivity parameter.  The ``typical_scale`` method computes the threshold
    as ``k * median(|signal|)``, which naturally adapts to the magnitude of each
    signal and works well across different derivative orders.
    """

    residual_method: Literal["mad", "percentile", "typical_scale"] = "mad"
    residual_k: float = 3.0
    residual_percentile: float = 95.0

    acceleration_method: Literal["mad", "percentile", "typical_scale"] = "mad"
    acceleration_k: float = 3.0
    acceleration_percentile: float = 95.0

    jerk_method: Literal["mad", "percentile", "typical_scale"] = "mad"
    jerk_k: float = 3.0
    jerk_percentile: float = 95.0

    dim_wise: bool = False


def extract_trend(
    data: np.ndarray,
    median_windows: list[int] | None = None,
    sg_window: int = 15,
    sg_order: int = 3,
) -> np.ndarray:
    """Extract smoothed long-term trend via cascaded median filtering + Savitzky-Golay.

    Args:
        data: Input trajectory, shape (T,) or (T, D).
        median_windows: Window sizes for cascaded median filtering (applied sequentially).
        sg_window: Window size for Savitzky-Golay filter (must be odd).
        sg_order: Polynomial order for Savitzky-Golay filter.

    Returns:
        Smoothed trend with same shape as input.
    """
    if median_windows is None:
        median_windows = [5, 11, 21]

    trend = np.asarray(data, dtype=np.float64)
    original_shape = trend.shape
    if trend.ndim == 1:
        trend = trend.reshape(-1, 1)

    # Cascaded median filtering
    for w in median_windows:
        if w >= 3:
            for d in range(trend.shape[1]):
                trend[:, d] = medfilt(trend[:, d], kernel_size=w)

    # Savitzky-Golay smoothing
    sg_win = min(sg_window, trend.shape[0] if trend.shape[0] % 2 == 1 else trend.shape[0] - 1)
    sg_win = max(sg_win, sg_order + 1 if (sg_order + 1) % 2 == 1 else sg_order + 2)
    if sg_win >= sg_order + 2 and trend.shape[0] > sg_win:
        for d in range(trend.shape[1]):
            trend[:, d] = savgol_filter(trend[:, d], window_length=sg_win, polyorder=sg_order)

    return trend.reshape(original_shape)


def compute_adaptive_threshold(
    signal: np.ndarray,
    method: Literal["typical_scale", "mad", "percentile"] = "typical_scale",
    k: float = 3.0,
    percentile: float = 95.0,
) -> float:
    """Compute an adaptive threshold for a deviation signal.

    Args:
        signal: Deviation values (1-D).
        method:
            - 'typical_scale': threshold = k * median(|signal|).
            - 'mad': threshold = median(signal) + k * MAD(signal).
            - 'percentile': threshold = given percentile of signal.
        k: Multiplier for typical_scale or MAD.
        percentile: Percentile to use (0-100).

    Returns:
        Scalar threshold value.
    """
    if method == "typical_scale":
        return float(k * np.median(np.abs(signal)))
    elif method == "mad":
        med = np.median(signal)
        mad = np.median(np.abs(signal - med))
        return float(med + k * mad)
    elif method == "percentile":
        return float(np.percentile(signal, percentile))
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_deviation_signals(
    data: np.ndarray,
    smoothed: np.ndarray,
    fps: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute residual, acceleration, and jerk deviation signals.

    Args:
        data: Raw trajectory, shape (T, D).
        smoothed: Smoothed trend, shape (T, D).
        fps: Frames per second (for physical scaling).

    Returns:
        Dict with keys:
            - 'residual': per-dim absolute residual, shape (T, D).
            - 'residual_magnitude': per-frame residual norm, shape (T,).
            - 'acceleration': per-frame acceleration magnitude, shape (T,).
            - 'jerk': per-frame jerk magnitude, shape (T,).
        Shorter signals are NaN-padded at edges.
    """
    data = np.asarray(data, dtype=np.float64)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    T, D = data.shape

    # Residual
    residual = np.abs(data - smoothed)
    residual_mag = np.linalg.norm(residual, axis=1) if D > 1 else residual.flatten()

    # Acceleration: second-order central difference
    acc_raw = np.full_like(data, np.nan)
    if T >= 3:
        acc_raw[1:-1] = (data[2:] - 2 * data[1:-1] + data[:-2]) * (fps**2)
    acc_mag = np.linalg.norm(acc_raw, axis=1) if D > 1 else acc_raw.flatten()

    # Jerk: third-order central difference
    jerk_raw = np.full_like(data, np.nan)
    if T >= 4:
        jerk_raw[1:-2] = (data[3:] - 3 * data[2:-1] + 3 * data[1:-2] - data[:-3]) * (fps**3)
    jerk_mag = np.linalg.norm(jerk_raw, axis=1) if D > 1 else jerk_raw.flatten()

    return {
        "residual": residual,
        "residual_magnitude": residual_mag,
        "acceleration": acc_mag,
        "jerk": jerk_mag,
    }


def detect_sudden_changes(
    data: np.ndarray,
    smoothed: np.ndarray,
    fps: float = 1.0,
    threshold_cfg: ThresholdConfig | None = None,
) -> dict:
    """Detect sudden changes in trajectory using multi-level deviation signals.

    A frame is flagged as a sudden change only when the residual, acceleration,
    and jerk ALL exceed their respective adaptive thresholds simultaneously.

    Args:
        data: Raw trajectory, shape (T, D).
        smoothed: Smoothed trend, shape (T, D).
        fps: Frames per second.
        threshold_cfg: Threshold configuration.

    Returns:
        Dict with keys:
            - 'events': boolean array of shape (T,) marking sudden changes.
            - 'residual_deviation': per-frame residual magnitude (T,).
            - 'acceleration_deviation': per-frame acceleration magnitude (T,).
            - 'jerk_deviation': per-frame jerk magnitude (T,).
            - 'residual_threshold': threshold value used.
            - 'acceleration_threshold': threshold value used.
            - 'jerk_threshold': threshold value used.
    """
    if threshold_cfg is None:
        threshold_cfg = ThresholdConfig()

    data = np.asarray(data, dtype=np.float64)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    T, D = data.shape

    dev = compute_deviation_signals(data, smoothed, fps)
    residual = dev["residual"]
    residual_mag = dev["residual_magnitude"]
    acc = dev["acceleration"]
    jerk = dev["jerk"]

    # Compute thresholds
    if threshold_cfg.dim_wise and D > 1:
        res_thresh = np.array([
            compute_adaptive_threshold(residual[:, d], threshold_cfg.residual_method,
                                       threshold_cfg.residual_k, threshold_cfg.residual_percentile)
            for d in range(D)
        ])
        res_event = np.any(residual > res_thresh, axis=1)
    else:
        res_thresh = compute_adaptive_threshold(
            residual_mag, threshold_cfg.residual_method,
            threshold_cfg.residual_k, threshold_cfg.residual_percentile,
        )
        res_event = residual_mag > res_thresh

    acc_valid = ~np.isnan(acc)
    if np.any(acc_valid):
        acc_thresh = compute_adaptive_threshold(
            acc[acc_valid], threshold_cfg.acceleration_method,
            threshold_cfg.acceleration_k, threshold_cfg.acceleration_percentile,
        )
        acc_event = np.where(acc_valid, acc > acc_thresh, False)
    else:
        acc_thresh = 0.0
        acc_event = np.zeros(T, dtype=bool)

    jerk_valid = ~np.isnan(jerk)
    if np.any(jerk_valid):
        jerk_thresh = compute_adaptive_threshold(
            jerk[jerk_valid], threshold_cfg.jerk_method,
            threshold_cfg.jerk_k, threshold_cfg.jerk_percentile,
        )
        jerk_event = np.where(jerk_valid, jerk > jerk_thresh, False)
    else:
        jerk_thresh = 0.0
        jerk_event = np.zeros(T, dtype=bool)

    events = res_event & acc_event & jerk_event

    return {
        "events": events,
        "residual_deviation": residual_mag,
        "acceleration_deviation": acc,
        "jerk_deviation": jerk,
        "residual_threshold": float(res_thresh) if np.ndim(res_thresh) == 0 else res_thresh.tolist(),
        "acceleration_threshold": float(acc_thresh),
        "jerk_threshold": float(jerk_thresh),
    }
