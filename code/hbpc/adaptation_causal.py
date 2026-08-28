from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from hbpc.metrics import events_from_binary
from hbpc.rrp import rank_effect


def fit_tau(errors_segment: np.ndarray, max_steps: int = 20, eps: float = 1e-8) -> dict[str, float | int]:
    """Fit e(t) ~= amplitude * exp(-t / tau) + residual to one error segment."""
    y = np.asarray(errors_segment, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    count = int(min(y.size, int(max_steps)))
    if count < 3:
        first = float(y[0]) if y.size else 0.0
        last = float(y[-1]) if y.size else 0.0
        return {"tau": float("inf"), "e_peak": first, "e_residual": last, "r_squared": 0.0, "n_points": count}

    y = y[:count]
    t = np.arange(count, dtype=np.float64)
    residual = max(float(y[-1]), 0.0)
    if float(np.polyfit(t, y, 1)[0]) >= 0:
        return _tau_result(float("inf"), y, residual, count, 0.0)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    residual0 = residual
    amplitude0 = max(float(y[0]) - residual0, eps)
    tau0 = max(float(count) / 4.0, 1.0)

    def model(x: np.ndarray, amplitude: float, tau: float, residual: float) -> np.ndarray:
        return amplitude * np.exp(-x / tau) + residual

    try:
        params, _ = curve_fit(
            model,
            t,
            y,
            p0=(amplitude0, tau0, residual0),
            bounds=([0.0, eps, 0.0], [max(y_max * 10.0, eps), float(count * 100), max(y_max * 2.0, eps)]),
            maxfev=10000,
        )
        amplitude, tau, residual = (float(params[0]), float(params[1]), float(params[2]))
        fitted = model(t, amplitude, tau, residual)
    except Exception:
        residual = max(min(y_min, residual0), 0.0)
        adjusted = y - residual + float(eps)
        if adjusted.size < 3 or np.allclose(adjusted, adjusted[0]):
            return _tau_result(float("inf"), y, residual, count, 0.0)
        slope, intercept = np.polyfit(t, np.log(adjusted), 1)
        if slope >= 0:
            return _tau_result(float("inf"), y, residual, count, 0.0)
        tau = float(-1.0 / slope)
        fitted = np.exp(intercept + slope * t) + residual

    if not np.isfinite(tau) or tau <= 0:
        return _tau_result(float("inf"), y, residual, count, 0.0)

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / max(ss_tot, eps))
    return _tau_result(tau, y, residual, count, r_squared)


def adaptation_features(
    errors: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray | None = None,
    max_steps: int = 20,
    high_fraction: float = 0.01,
    min_r_squared: float = 0.0,
) -> dict[str, float | int]:
    """Measure anomaly and normal-high recovery time constants."""
    err = np.asarray(errors, dtype=np.float64)
    if err.ndim != 2:
        raise ValueError("errors must have shape [time, channels]")
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    if err.shape[0] != label_arr.size:
        raise ValueError("errors and labels must have the same time length")
    score_arr = np.linalg.norm(np.nan_to_num(err, nan=0.0), axis=1) if scores is None else np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_arr.shape != label_arr.shape:
        raise ValueError("scores and labels must have the same shape")
    norms = np.linalg.norm(np.nan_to_num(err, nan=0.0), axis=1)

    anomaly_taus, anomaly_peak_ratios = _segment_tau_values(
        norms,
        events_from_binary(label_arr),
        max_steps=max_steps,
        min_r_squared=min_r_squared,
    )
    normal_high_events = _normal_high_events(
        score_arr,
        label_arr,
        high_fraction=high_fraction,
        max_steps=max_steps,
    )
    normal_taus, _normal_peak_ratios = _segment_tau_values(
        norms,
        normal_high_events,
        max_steps=max_steps,
        min_r_squared=min_r_squared,
    )

    anomaly_lengths = [stop - start + 1 for start, stop in events_from_binary(label_arr)]
    normal_lengths = [stop - start + 1 for start, stop in normal_high_events]
    anomaly_median = _finite_median(anomaly_taus)
    normal_median = _finite_median(normal_taus)
    return {
        "tau_anomaly_median": anomaly_median,
        "tau_anomaly_mean": _finite_mean(anomaly_taus),
        "tau_anomaly_n": int(len(anomaly_taus)),
        "tau_normal_median": normal_median,
        "tau_normal_mean": _finite_mean(normal_taus),
        "tau_normal_n": int(len(normal_taus)),
        "tau_ratio": float(anomaly_median / normal_median) if np.isfinite(anomaly_median) and np.isfinite(normal_median) and normal_median > 0 else float("inf"),
        "peak_residual_ratio_median": _finite_median(anomaly_peak_ratios),
        "anomaly_segment_mean_length": _finite_mean(anomaly_lengths),
        "anomaly_segment_median_length": _finite_median(anomaly_lengths),
        "normal_high_segment_mean_length": _finite_mean(normal_lengths),
        "normal_high_segment_median_length": _finite_median(normal_lengths),
        "anomaly_event_count": int(len(anomaly_lengths)),
        "normal_high_event_count": int(len(normal_lengths)),
    }


def compute_relaxation_ratio(scores: np.ndarray, k: int, eps: float = 1e-8) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    out = np.full(arr.size, np.nan, dtype=np.float64)
    for idx in range(0, max(arr.size - int(k), 0)):
        future = arr[idx + 1 : idx + int(k) + 1]
        finite = future[np.isfinite(future)]
        out[idx] = float(np.mean(finite) / (arr[idx] + eps)) if finite.size and np.isfinite(arr[idx]) else np.nan
    return out


def compute_rank_biserial(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    high_fraction: float = 0.01,
) -> float:
    """Rank-biserial separation between anomaly-high and normal-high relaxation ratios."""
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    if arr.shape != label_arr.shape:
        raise ValueError("scores and labels must have the same shape")
    relax = compute_relaxation_ratio(arr, k=k)
    finite = np.isfinite(arr) & np.isfinite(relax)
    if not finite.any():
        return 0.0
    normal_mask = finite & ~label_arr
    anomaly_mask = finite & label_arr
    if not normal_mask.any() or not anomaly_mask.any():
        return 0.0
    normal_idx = _top_fraction_indices(arr, normal_mask, high_fraction)
    anomaly_idx = _top_fraction_indices(arr, anomaly_mask, high_fraction)
    normal = relax[normal_idx]
    anomaly = relax[anomaly_idx]
    effect = rank_effect(normal, anomaly)
    return 0.0 if not np.isfinite(effect.rank_biserial) else float(effect.rank_biserial)


def _normal_high_events(
    scores: np.ndarray,
    labels: np.ndarray,
    high_fraction: float,
    max_steps: int,
) -> list[tuple[int, int]]:
    if not 0.0 < high_fraction < 1.0:
        raise ValueError("high_fraction must be in (0, 1)")
    normal = ~labels
    finite_normal = normal & np.isfinite(scores)
    if not finite_normal.any():
        return []
    selected = np.zeros(scores.size, dtype=bool)
    selected[_top_fraction_indices(scores, finite_normal, high_fraction)] = True
    segments: list[tuple[int, int]] = []
    for start, _stop in events_from_binary(selected):
        stop = start
        while stop + 1 < scores.size and stop - start + 1 < max_steps and not labels[stop + 1]:
            stop += 1
        if stop > start:
            segments.append((start, stop))
    return segments


def _top_fraction_indices(scores: np.ndarray, mask: np.ndarray, high_fraction: float) -> np.ndarray:
    idx = np.flatnonzero(mask & np.isfinite(scores))
    if idx.size == 0:
        return idx
    n_top = max(1, int(np.ceil(idx.size * float(high_fraction))))
    values = scores[idx]
    sorted_order = np.argsort(values, kind="mergesort")
    return idx[sorted_order[-n_top:]]


def _segment_tau_values(
    norms: np.ndarray,
    segments: list[tuple[int, int]],
    max_steps: int,
    min_r_squared: float,
) -> tuple[list[float], list[float]]:
    taus: list[float] = []
    peak_ratios: list[float] = []
    for start, stop in segments:
        segment = norms[start : stop + 1]
        result = fit_tau(segment, max_steps=max_steps)
        tau = float(result["tau"])
        r_squared = float(result["r_squared"])
        if np.isfinite(tau) and r_squared >= min_r_squared:
            taus.append(tau)
            residual = max(float(result["e_residual"]), 1e-8)
            peak_ratios.append(float(result["e_peak"]) / residual)
    return taus, peak_ratios


def _tau_result(tau: float, y: np.ndarray, residual: float, count: int, r_squared: float) -> dict[str, float | int]:
    return {
        "tau": float(tau),
        "e_peak": float(y[0]) if y.size else 0.0,
        "e_residual": float(residual),
        "r_squared": float(max(min(r_squared, 1.0), 0.0)),
        "n_points": int(count),
    }


def _finite_mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("inf")


def _finite_median(values) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("inf")
