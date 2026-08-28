from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def top_n_indices(scores: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.isfinite(values))
    if n <= 0 or valid.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((valid, -values[valid]))
    return valid[order[: min(int(n), order.size)]].astype(np.int64)


def shift_alarm_indices(indices: np.ndarray, shift: int, length: int) -> np.ndarray:
    shifted = np.asarray(indices, dtype=np.int64).reshape(-1) + int(shift)
    return shifted[(shifted >= 0) & (shifted < int(length))]


def jaccard_indices(first: Iterable[int], second: Iterable[int]) -> float:
    a = set(int(x) for x in first)
    b = set(int(x) for x in second)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def tail_score(residuals: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    scores = np.full(r.size, np.nan, dtype=np.float64)
    for t in range(0, r.size - int(k)):
        scores[t] = _finite_mean(r[t + 1 : t + int(k) + 1])
    return scores


def mars_abs_score(residuals: np.ndarray, k: int, alpha: float) -> np.ndarray:
    forward = tail_score(residuals, k)
    backward = _backward_mean(residuals, k)
    scores = forward + float(alpha) * (forward - backward)
    return np.maximum(scores, 0.0)


def mars_rel_score(residuals: np.ndarray, k: int, alpha: float, eps: float = 1e-8) -> np.ndarray:
    forward = tail_score(residuals, k)
    backward = _backward_mean(residuals, k)
    momentum = (forward - backward) / (forward + backward + float(eps))
    scores = forward * (1.0 + float(alpha) * momentum)
    return np.maximum(scores, 0.0)


def ewma_score(residuals: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    scores = np.empty(r.size, dtype=np.float64)
    if r.size == 0:
        return scores
    scores[0] = r[0] if np.isfinite(r[0]) else 0.0
    for idx in range(1, r.size):
        value = r[idx] if np.isfinite(r[idx]) else 0.0
        scores[idx] = float(alpha) * value + (1.0 - float(alpha)) * scores[idx - 1]
    return scores


def _backward_mean(residuals: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    values = np.full(r.size, np.nan, dtype=np.float64)
    for t in range(int(k) - 1, r.size):
        values[t] = _finite_mean(r[t - int(k) + 1 : t + 1])
    return values


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")
