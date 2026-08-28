from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hbpc.calibration import eventize


def miss_rate(pred: np.ndarray, labels: np.ndarray, min_gap: int = 0) -> float:
    pred_arr, label_arr = _aligned_bool_arrays(pred, labels)
    attacks = eventize(label_arr, min_gap=min_gap)
    if not attacks:
        return 0.0
    hits = sum(bool(pred_arr[start : stop + 1].any()) for start, stop in attacks)
    return float(1.0 - hits / len(attacks))


def mttd(pred: np.ndarray, labels: np.ndarray, min_gap: int = 0) -> float:
    pred_arr, label_arr = _aligned_bool_arrays(pred, labels)
    delays: list[int] = []
    for start, stop in eventize(label_arr, min_gap=min_gap):
        hits = np.flatnonzero(pred_arr[start : stop + 1])
        if hits.size:
            delays.append(int(hits[0]))
    return float(np.mean(delays)) if delays else float("nan")


def wr_off(
    pred: np.ndarray,
    labels: np.ndarray,
    budget: float,
    fs: float = 1.0,
    min_gap: int = 0,
) -> float:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")

    pred_arr, label_arr = _aligned_bool_arrays(pred, labels)
    alarm_events = eventize(pred_arr, min_gap=min_gap)
    attack_events = eventize(label_arr, min_gap=min_gap)
    off_events = [
        event for event in alarm_events if not _overlaps_any(event, attack_events)
    ]

    off_points = int((~label_arr).sum())
    off_hours = off_points / (3600.0 * fs)
    if off_hours == 0:
        return 0.0 if not off_events else float("inf")
    return float((len(off_events) / off_hours) / budget)


def wr_total(
    pred: np.ndarray,
    budget: float,
    fs: float = 1.0,
    min_gap: int = 0,
) -> float:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")

    pred_arr = np.asarray(pred).astype(bool).reshape(-1)
    if pred_arr.size == 0:
        raise ValueError("pred must not be empty")
    total_hours = pred_arr.size / (3600.0 * fs)
    events = eventize(pred_arr, min_gap=min_gap)
    return float((len(events) / total_hours) / budget)


def budget_violation_rate(workload_ratios: Sequence[float]) -> float:
    values = np.asarray(workload_ratios, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values > 1.0))


def threshold_fold_cv(thresholds: Sequence[float], eps: float = 1e-12) -> float:
    values = np.asarray(thresholds, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.std(values) / (abs(np.mean(values)) + eps))


def _aligned_bool_arrays(
    pred: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred).astype(bool).reshape(-1)
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    if pred_arr.shape != label_arr.shape:
        raise ValueError("pred and labels must have the same shape")
    return pred_arr, label_arr


def _overlaps_any(
    event: tuple[int, int],
    candidates: Sequence[tuple[int, int]],
) -> bool:
    start, stop = event
    return any(start <= other_stop and stop >= other_start for other_start, other_stop in candidates)
