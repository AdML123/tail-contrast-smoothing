from __future__ import annotations

import numpy as np


def robust_zscore(
    scores: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    ref = np.asarray(reference, dtype=float).reshape(-1)
    valid_ref = ref[np.isfinite(ref)]
    if valid_ref.size == 0:
        raise ValueError("reference must contain at least one finite value")

    median = float(np.median(valid_ref))
    q75, q25 = np.percentile(valid_ref, [75, 25])
    iqr = float(q75 - q25)
    scale = max(iqr, eps)
    z = (scores_arr - median) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    return np.maximum(z, 0.0)


def median_smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()

    radius = window // 2
    padded = np.pad(arr, (radius, radius), mode="edge")
    smoothed = np.empty_like(arr, dtype=float)
    for idx in range(arr.size):
        window_values = padded[idx : idx + window]
        finite = window_values[np.isfinite(window_values)]
        smoothed[idx] = float(np.median(finite)) if finite.size else 0.0
    return smoothed


def local_peaks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        return np.array([], dtype=int)
    finite = np.isfinite(arr)
    mask = finite[1:-1] & (arr[1:-1] > arr[:-2]) & (arr[1:-1] >= arr[2:])
    return np.flatnonzero(mask) + 1


def apply_refractory(
    peaks: np.ndarray,
    values: np.ndarray,
    refractory: int = 0,
) -> np.ndarray:
    if refractory < 0:
        raise ValueError("refractory must be non-negative")
    peak_idx = np.asarray(peaks, dtype=int).reshape(-1)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if peak_idx.size == 0:
        return peak_idx
    if np.any((peak_idx < 0) | (peak_idx >= arr.size)):
        raise ValueError("peaks contain out-of-range indices")

    order = sorted(peak_idx.tolist())
    selected: list[int] = []
    blocked_until = -1
    for idx in order:
        if idx > blocked_until:
            selected.append(idx)
            blocked_until = idx + refractory
    return np.array(selected, dtype=int)


def top_n_peak_events(
    values: np.ndarray,
    n: int,
    refractory: int = 0,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    arr = np.asarray(values, dtype=float).reshape(-1)
    peaks = local_peaks(arr)
    if peaks.size == 0:
        return np.zeros(arr.size, dtype=np.int64)

    order = sorted(peaks.tolist(), key=lambda idx: (-arr[idx], idx))[:n]
    selected = apply_refractory(np.array(order, dtype=int), arr, refractory=refractory)
    pred = np.zeros(arr.size, dtype=np.int64)
    pred[selected] = 1
    return pred


def quantile_peak_events(
    values: np.ndarray,
    quantile: float,
    refractory: int = 0,
) -> np.ndarray:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(values, dtype=float).reshape(-1)
    peaks = local_peaks(arr)
    if peaks.size == 0:
        return np.zeros(arr.size, dtype=np.int64)

    threshold = float(np.quantile(arr[peaks], quantile))
    candidates = peaks[arr[peaks] > threshold]
    selected = apply_refractory(candidates, arr, refractory=refractory)
    pred = np.zeros(arr.size, dtype=np.int64)
    pred[selected] = 1
    return pred
