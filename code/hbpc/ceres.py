from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_scores(scores: np.ndarray, center: float | None = None, scale: float | None = None) -> tuple[np.ndarray, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.size, dtype=np.float64), 1.0
    used_center = float(np.median(finite)) if center is None else float(center)
    used_scale = _robust_scale(finite) if scale is None else float(scale)
    if not np.isfinite(used_scale) or used_scale <= 0:
        used_scale = 1.0
    normalized = (np.nan_to_num(values, nan=used_center, posinf=used_center, neginf=used_center) - used_center) / used_scale
    return np.maximum(normalized, 0.0), used_scale


def robust_location_scale(scores: np.ndarray) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = _robust_scale(finite)
    return center, scale if scale > 0 and np.isfinite(scale) else 1.0


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    bin_edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if bin_edges.size < 2:
        return np.zeros(vals.size, dtype=np.int64)
    bins = np.searchsorted(bin_edges[1:-1], vals, side="right")
    return np.clip(bins, 0, bin_edges.size - 2).astype(np.int64)


def build_tail_envelopes(
    z: np.ndarray,
    k: int,
    bin_probs: Sequence[float],
    q_upper: float,
    min_bin_count: int,
) -> tuple[dict[str, np.ndarray | float | int], list[dict[str, object]]]:
    if k <= 0:
        raise ValueError("k must be positive")
    values = np.asarray(z, dtype=np.float64).reshape(-1)
    peaks, tails, steps = _peak_tail_samples(values, int(k))
    if peaks.size == 0:
        env = _empty_envelope(int(k), float(q_upper), int(min_bin_count))
        return env, [_diagnostic_row(0, [0], 0, False, 0.0)]

    probs = np.asarray(bin_probs, dtype=np.float64).reshape(-1)
    if probs.size < 2:
        probs = np.array([0.0, 1.0], dtype=np.float64)
    probs = np.clip(np.sort(probs), 0.0, 1.0)
    probs[0] = 0.0
    probs[-1] = 1.0
    edges = np.quantile(peaks, probs)
    raw_bins = assign_bins(peaks, edges)
    groups = _merge_sparse_bins(raw_bins, edges.size - 1, int(min_bin_count))

    tail_quantiles: list[float] = []
    step_quantiles: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    raw_to_effective = np.zeros(edges.size - 1, dtype=np.int64)
    for effective, (raw_members, sample_indices) in enumerate(groups):
        idx = np.asarray(sample_indices, dtype=np.int64)
        if idx.size == 0:
            tail_q = 0.0
            step_q = np.zeros(int(k), dtype=np.float64)
        else:
            tail_q = float(np.quantile(tails[idx], float(q_upper)))
            step_q = np.quantile(steps[idx], float(q_upper), axis=0).astype(np.float64)
        for raw_bin in raw_members:
            raw_to_effective[int(raw_bin)] = int(effective)
        tail_quantiles.append(tail_q)
        step_quantiles.append(step_q)
        diagnostics.append(_diagnostic_row(effective, raw_members, int(idx.size), len(raw_members) > 1, tail_q))

    env: dict[str, np.ndarray | float | int] = {
        "bin_edges": edges.astype(np.float64),
        "raw_to_effective": raw_to_effective.astype(np.int64),
        "tail_quantiles": np.asarray(tail_quantiles, dtype=np.float64),
        "step_quantiles": np.vstack(step_quantiles).astype(np.float64),
        "q_upper": float(q_upper),
        "min_bin_count": int(min_bin_count),
    }
    return env, diagnostics


def ceres_lite_score(z: np.ndarray, env: dict[str, np.ndarray | float | int], k: int) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64).reshape(-1)
    scores = np.full(values.size, np.nan, dtype=np.float64)
    count = max(0, values.size - int(k))
    if count == 0:
        return scores
    peaks = values[:count]
    future = _future_matrix(values, int(k), count)
    with np.errstate(invalid="ignore"):
        tails = np.nanmean(future, axis=1)
    effective = _effective_bins(peaks, env)
    thresholds = np.asarray(env["tail_quantiles"], dtype=np.float64)[effective]
    valid = np.isfinite(peaks) & np.isfinite(tails)
    scores[:count] = np.where(
        valid,
        np.log1p(np.maximum(peaks, 0.0)) * np.maximum(tails - thresholds, 0.0),
        np.nan,
    )
    return scores


def ceres_envelope_score(z: np.ndarray, env: dict[str, np.ndarray | float | int], k: int) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64).reshape(-1)
    scores = np.full(values.size, np.nan, dtype=np.float64)
    step_quantiles = np.asarray(env["step_quantiles"], dtype=np.float64)
    count = max(0, values.size - int(k))
    if count == 0:
        return scores
    peaks = values[:count]
    future = _future_matrix(values, int(k), count)
    effective = _effective_bins(peaks, env)
    excess = np.maximum(np.nan_to_num(future, nan=0.0) - step_quantiles[effective], 0.0)
    valid = np.isfinite(peaks)
    scores[:count] = np.where(
        valid,
        np.log1p(np.maximum(peaks, 0.0)) * np.sum(excess, axis=1),
        np.nan,
    )
    return scores


def _robust_scale(values: np.ndarray) -> float:
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    if np.isfinite(iqr) and iqr > 0:
        return iqr / 1.349
    mad = float(np.median(np.abs(values - np.median(values))))
    return mad / 0.6745 if mad > 0 and np.isfinite(mad) else float(np.std(values))


def _peak_tail_samples(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peaks: list[float] = []
    tails: list[float] = []
    steps: list[np.ndarray] = []
    for t in range(0, max(0, values.size - k)):
        future = values[t + 1 : t + k + 1]
        peak = values[t]
        if future.size != k or not np.isfinite(peak) or not np.all(np.isfinite(future)):
            continue
        peaks.append(float(peak))
        tails.append(_finite_mean(future))
        steps.append(future.astype(np.float64))
    if not peaks:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.empty((0, k), dtype=np.float64)
    return np.asarray(peaks), np.asarray(tails), np.vstack(steps)


def _merge_sparse_bins(raw_bins: np.ndarray, n_bins: int, min_bin_count: int) -> list[tuple[list[int], np.ndarray]]:
    groups: list[tuple[list[int], np.ndarray]] = [
        ([idx], np.flatnonzero(raw_bins == idx).astype(np.int64)) for idx in range(max(1, int(n_bins)))
    ]
    min_count = max(1, int(min_bin_count))
    idx = len(groups) - 1
    while idx > 0:
        raw_members, sample_indices = groups[idx]
        if sample_indices.size < min_count:
            left_members, left_indices = groups[idx - 1]
            merged_members = [*left_members, *raw_members]
            merged_indices = np.sort(np.concatenate([left_indices, sample_indices])).astype(np.int64)
            groups[idx - 1] = (merged_members, merged_indices)
            groups.pop(idx)
        idx -= 1
    if len(groups) > 1 and groups[0][1].size < min_count:
        raw_members, sample_indices = groups[0]
        right_members, right_indices = groups[1]
        groups[1] = ([*raw_members, *right_members], np.sort(np.concatenate([sample_indices, right_indices])).astype(np.int64))
        groups.pop(0)
    return groups


def _empty_envelope(k: int, q_upper: float, min_bin_count: int) -> dict[str, np.ndarray | float | int]:
    return {
        "bin_edges": np.array([0.0, 1.0], dtype=np.float64),
        "raw_to_effective": np.array([0], dtype=np.int64),
        "tail_quantiles": np.array([0.0], dtype=np.float64),
        "step_quantiles": np.zeros((1, k), dtype=np.float64),
        "q_upper": float(q_upper),
        "min_bin_count": int(min_bin_count),
    }


def _diagnostic_row(
    effective_bin: int,
    raw_bins: Sequence[int],
    sample_count: int,
    merged: bool,
    tail_quantile: float,
) -> dict[str, object]:
    return {
        "effective_bin": int(effective_bin),
        "raw_bins": ";".join(str(int(x)) for x in raw_bins),
        "sample_count": int(sample_count),
        "merged_from": bool(merged),
        "tail_quantile": float(tail_quantile),
    }


def _effective_bin(value: float, env: dict[str, np.ndarray | float | int]) -> int:
    raw_bin = assign_bins(np.array([value], dtype=np.float64), np.asarray(env["bin_edges"], dtype=np.float64))[0]
    mapping = np.asarray(env["raw_to_effective"], dtype=np.int64)
    return int(mapping[min(int(raw_bin), mapping.size - 1)])


def _effective_bins(values: np.ndarray, env: dict[str, np.ndarray | float | int]) -> np.ndarray:
    raw_bins = assign_bins(np.asarray(values, dtype=np.float64), np.asarray(env["bin_edges"], dtype=np.float64))
    mapping = np.asarray(env["raw_to_effective"], dtype=np.int64)
    return mapping[np.minimum(raw_bins, mapping.size - 1)]


def _future_matrix(values: np.ndarray, k: int, count: int) -> np.ndarray:
    return np.stack([values[offset : offset + count] for offset in range(1, k + 1)], axis=1)


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")
