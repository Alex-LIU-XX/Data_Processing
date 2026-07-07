"""Stage 2: State-Action Trend Alignment.

Cross-correlation delay detection and Directional Agreement (DA) check
for identifying temporal misalignment between action and state trajectories.
"""

import dataclasses

import numpy as np
from scipy import stats


@dataclasses.dataclass
class AlignmentConfig:
    """Configuration for trend alignment analysis.

    Attributes:
        max_lag: Maximum frames to test for lag (both directions).
        correlation_method: How to aggregate per-dim correlations.
            "mean" — average correlation across dims.
            "median" — median correlation across dims.
        da_tolerance: Absolute difference below which direction is considered "no change".
        suspicious_lag_ratio: If |optimal_lag| > max_lag * this_ratio, flag as suspicious.
        suspicious_da_threshold: DA score below this threshold flags the episode.
    """

    max_lag: int = 50
    correlation_method: str = "mean"
    da_tolerance: float = 1e-6
    suspicious_lag_ratio: float = 0.8
    suspicious_da_threshold: float = 0.6


# ============================================================================
# Cross-correlation delay detection
# ============================================================================


def _pearson_at_lag(source: np.ndarray, target: np.ndarray, lag: int) -> np.ndarray:
    """Compute per-dim Pearson correlation at a given lag.

    Args:
        source: (T, D) array (action).
        target: (T, D) array (state).
        lag: Integer shift.  Positive = source is shifted right (source leads).

    Returns:
        Per-dim correlation coefficients, shape (D,).
    """
    T, D = source.shape
    if lag >= 0:
        src = source[lag:]
        tgt = target[:T - lag]
    else:
        src = source[:T + lag]
        tgt = target[-lag:]

    if len(src) < 3:
        return np.full(D, np.nan)

    corrs = np.array([
        stats.pearsonr(src[:, d], tgt[:, d])[0] if np.std(src[:, d]) > 1e-12 and np.std(tgt[:, d]) > 1e-12
        else 0.0
        for d in range(D)
    ])
    return corrs


def compute_optimal_lag(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 50,
    correlation_method: str = "mean",
) -> dict:
    """Compute optimal time lag between source (action) and target (state) via cross-correlation.

    Source is shifted relative to target.  A positive lag means source *leads*
    (source values happen earlier in time).

    Args:
        source: (T, D) array (action).
        target: (T, D) array (state).
        max_lag: Maximum frames to test.
        correlation_method: "mean" or "median" for aggregating per-dim correlations.

    Returns:
        Dict with:
            - lags: 1-D array of tested lags.
            - correlations: array of shape (L,) — aggregated correlation at each lag.
            - per_dim_correlations: array of shape (L, D) — per-dim correlation at each lag.
            - optimal_lag: lag that maximizes aggregated correlation.
            - optimal_correlation: max aggregated correlation value.
            - per_dim_optimal_lags: optimal lag per dimension (D,).
            - per_dim_optimal_corrs: optimal correlation per dimension (D,).
            - is_suspicious: True if lag violates causality or is unreasonably large.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T, D = source.shape

    lags = np.arange(-max_lag, max_lag + 1)
    per_dim_corrs = np.full((len(lags), D), np.nan)

    for i, lag in enumerate(lags):
        per_dim_corrs[i] = _pearson_at_lag(source, target, lag)

    # Aggregate across dims
    if correlation_method == "median":
        agg_corr = np.nanmedian(per_dim_corrs, axis=1)
    else:
        agg_corr = np.nanmean(per_dim_corrs, axis=1)

    # Overall optimal lag
    valid = ~np.isnan(agg_corr)
    if not np.any(valid):
        agg_corr[:] = 0.0
        valid = np.ones_like(agg_corr, dtype=bool)

    optimal_idx = np.argmax(agg_corr[valid])
    optimal_lag = int(lags[valid][optimal_idx])
    optimal_correlation = float(agg_corr[valid][optimal_idx])

    # Per-dim optimal lags
    per_dim_optimal_lags = np.full(D, np.nan, dtype=np.float64)
    per_dim_optimal_corrs = np.full(D, np.nan, dtype=np.float64)
    for d in range(D):
        pd_valid = ~np.isnan(per_dim_corrs[:, d])
        if np.any(pd_valid):
            pd_idx = np.argmax(per_dim_corrs[pd_valid, d])
            per_dim_optimal_lags[d] = lags[pd_valid][pd_idx]
            per_dim_optimal_corrs[d] = per_dim_corrs[pd_valid, d][pd_idx]

    # Suspicion checks:
    #   lag > 0  → source (action) lags target (state)  → violates causality  → suspicious
    #   lag <= 0 → source leads or is in sync            → OK
    #   |lag| too large or per-dim inconsistencies       → suspicious
    is_suspicious = bool(
        optimal_lag > 0
        or abs(optimal_lag) > max_lag * 0.8
        or np.nanstd(per_dim_optimal_lags) > max_lag * 0.5
    )

    return {
        "lags": lags,
        "correlations": agg_corr,
        "per_dim_correlations": per_dim_corrs,
        "optimal_lag": optimal_lag,
        "optimal_correlation": optimal_correlation,
        "per_dim_optimal_lags": per_dim_optimal_lags.tolist(),
        "per_dim_optimal_corrs": per_dim_optimal_corrs.tolist(),
        "is_suspicious": is_suspicious,
    }


# ============================================================================
# Directional Agreement (DA) Check
# ============================================================================


def directional_agreement(
    aligned_source: np.ndarray,
    target: np.ndarray,
    tolerance: float = 1e-6,
) -> dict:
    """Compute Directional Agreement (DA) between time-aligned source and target.

    Checks that first-order differences have the same sign (both moving in the
    same direction).

    Args:
        aligned_source: (T, D) array — source shifted by optimal lag.
        target: (T, D) array — target (same length as aligned_source).
        tolerance: Absolute diff below which direction is "no change" (considered agreeing).

    Returns:
        Dict with:
            - da_per_frame: (T-1, D) boolean — agreement per frame per dim.
            - da_per_dim: (D,) — fraction of agreeing frames per dimension.
            - da_overall: float — fraction of agreeing frames across all dims.
            - disagreement_frames: 1-D array of frame indices where fewer than half of dims agree.
    """
    aligned_source = np.asarray(aligned_source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T, D = aligned_source.shape

    src_diff = np.diff(aligned_source, axis=0)
    tgt_diff = np.diff(target, axis=0)

    both_flat = (np.abs(src_diff) < tolerance) & (np.abs(tgt_diff) < tolerance)
    same_sign = np.sign(src_diff) == np.sign(tgt_diff)
    agree = both_flat | same_sign

    da_per_dim = np.mean(agree, axis=0)
    da_overall = float(np.mean(agree))

    # Frames where fewer than half the dims agree
    n_agree = np.sum(agree, axis=1)
    disagree_mask = n_agree < (D / 2.0)
    disagreement_frames = np.where(disagree_mask)[0]

    return {
        "da_per_frame": agree,
        "da_per_dim": da_per_dim.tolist(),
        "da_overall": da_overall,
        "disagreement_frames": disagreement_frames.tolist(),
    }


# ============================================================================
# Combined alignment analysis
# ============================================================================


def align_signals(source: np.ndarray, target: np.ndarray, lag: int):
    """Align source to target by shifting source by ``lag`` frames.

    Args:
        source: (T, D) array.
        target: (T, D) array.
        lag: Shift to apply (positive = source is shifted right).

    Returns:
        (aligned_source, aligned_target) — trimmed to overlapping region.
    """
    T, D = source.shape
    if lag >= 0:
        return source[lag:].copy(), target[:T - lag].copy()
    else:
        return source[:T + lag].copy(), target[-lag:].copy()


def analyze_trend_alignment(
    actions: np.ndarray,
    states: np.ndarray,
    cfg: AlignmentConfig | None = None,
) -> dict:
    """Complete Stage 2: State-Action Trend Alignment analysis.

    Steps:
        1. Compute optimal lag via cross-correlation (action → state).
        2. Align action to state using optimal lag.
        3. Compute Directional Agreement on aligned data.
        4. Flag suspicious episodes.

    Args:
        actions: (T, D) action trajectory.
        states: (T, D) state trajectory.
        cfg: Alignment configuration.

    Returns:
        Dict with all results from lag detection + DA + flags.
    """
    if cfg is None:
        cfg = AlignmentConfig()

    # 1. Lag detection
    lag_result = compute_optimal_lag(
        actions, states,
        max_lag=cfg.max_lag,
        correlation_method=cfg.correlation_method,
    )

    # 2. Align
    aligned_action, aligned_state = align_signals(actions, states, lag_result["optimal_lag"])

    # 3. Directional Agreement
    da_result = directional_agreement(
        aligned_action, aligned_state,
        tolerance=cfg.da_tolerance,
    )

    # 4. Suspicious flags
    is_suspicious_lag = lag_result["is_suspicious"]
    is_suspicious_da = da_result["da_overall"] < cfg.suspicious_da_threshold
    is_suspicious = is_suspicious_lag or is_suspicious_da

    return {
        "lag": lag_result,
        "da": da_result,
        "aligned_action": aligned_action,
        "aligned_state": aligned_state,
        "is_suspicious": is_suspicious,
        "is_suspicious_lag": is_suspicious_lag,
        "is_suspicious_da": is_suspicious_da,
    }
