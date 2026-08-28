from __future__ import annotations

import numpy as np


def build_scar_calibration(
    residuals: np.ndarray,
    k: int,
    n_bins: int,
    q_upper: float,
    n_ref: int,
    min_bin_count: int = 30,
    labels: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray | float | int], list[dict[str, object]]]:
    if k <= 0:
        raise ValueError("k must be positive")
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    cal_labels = None if labels is None else np.asarray(labels).astype(bool).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        edges = np.array([-np.inf, np.inf], dtype=np.float64)
    else:
        edges = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf

    tail = _tail_mean(values, int(k))
    bins = assign_scar_bins(values, edges)
    q_values = np.full(int(n_bins), np.nan, dtype=np.float64)
    weights = np.zeros(int(n_bins), dtype=np.float64)
    counts = np.zeros(int(n_bins), dtype=np.int64)
    diagnostics: list[dict[str, object]] = []

    for bin_id in range(int(n_bins)):
        mask = (bins == bin_id) & np.isfinite(tail)
        bin_tail = tail[mask]
        counts[bin_id] = int(bin_tail.size)
        if bin_tail.size >= int(min_bin_count):
            q_values[bin_id] = float(np.quantile(bin_tail, float(q_upper)))
            weights[bin_id] = min(1.0, float(bin_tail.size) / max(1, int(n_ref)))
        label_normal_count = 0
        label_anomaly_count = 0
        if cal_labels is not None and cal_labels.size == values.size:
            label_normal_count = int((mask & ~cal_labels).sum())
            label_anomaly_count = int((mask & cal_labels).sum())
        diagnostics.append(
            {
                "bin_id": int(bin_id),
                "bin_low": float(edges[bin_id]),
                "bin_high": float(edges[bin_id + 1]),
                "n_b": int(bin_tail.size),
                "Q_b": float(q_values[bin_id]) if np.isfinite(q_values[bin_id]) else np.nan,
                "w": float(weights[bin_id]),
                "label_normal_count": label_normal_count,
                "label_anomaly_count": label_anomaly_count,
            }
        )

    calibration: dict[str, np.ndarray | float | int] = {
        "bin_edges": edges.astype(np.float64),
        "q_values": q_values,
        "weights": weights,
        "counts": counts,
        "k": int(k),
        "q_upper": float(q_upper),
        "n_ref": int(n_ref),
        "min_bin_count": int(min_bin_count),
    }
    return calibration, diagnostics


def scar_score(residuals: np.ndarray, calibration: dict[str, np.ndarray | float | int], k: int, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    tail = _tail_mean(values, int(k))
    scores = tail.copy()
    bins = assign_scar_bins(values, np.asarray(calibration["bin_edges"], dtype=np.float64))
    q_values = np.asarray(calibration["q_values"], dtype=np.float64)
    weights = np.asarray(calibration["weights"], dtype=np.float64)
    valid = np.isfinite(tail)
    valid &= bins >= 0
    valid &= bins < q_values.size
    q = np.full(values.size, np.nan, dtype=np.float64)
    w = np.zeros(values.size, dtype=np.float64)
    q[valid] = q_values[bins[valid]]
    w[valid] = weights[bins[valid]]
    usable = valid & np.isfinite(q) & (w > 0)
    enhancement = np.zeros(values.size, dtype=np.float64)
    enhancement[usable] = np.maximum((tail[usable] - q[usable]) / (tail[usable] + float(eps)), 0.0)
    scores[usable] = tail[usable] * (1.0 + w[usable] * enhancement[usable])
    return scores


def assign_scar_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    bin_edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if bin_edges.size < 2:
        return np.zeros(vals.size, dtype=np.int64)
    bins = np.searchsorted(bin_edges[1:-1], vals, side="right")
    return np.clip(bins, 0, bin_edges.size - 2).astype(np.int64)


def _tail_mean(values: np.ndarray, k: int) -> np.ndarray:
    scores = np.full(values.size, np.nan, dtype=np.float64)
    count = max(0, values.size - int(k))
    if count == 0:
        return scores
    future = np.stack([values[offset : offset + count] for offset in range(1, int(k) + 1)], axis=1)
    finite = np.isfinite(future)
    totals = np.where(finite, future, 0.0).sum(axis=1)
    counts = finite.sum(axis=1)
    scores[:count] = np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)
    return scores
