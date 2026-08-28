from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from hbpc.metrics import detection_delay, events_from_binary, f1


def paired_bootstrap_ci(differences: np.ndarray, n_boot: int = 10_000, seed: int = 47) -> dict[str, float | int | str]:
    values = np.asarray(differences, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("at least one finite matched-pair difference is required")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(n_boot, values.size))
    estimates = np.mean(values[indices], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"unit": "matched_event_pair", "n": int(values.size), "estimate": float(np.mean(values)), "ci_low": float(low), "ci_high": float(high)}


def cluster_bootstrap_metric(cluster_ids: np.ndarray, values: np.ndarray, statistic: Callable[[np.ndarray], float], n_boot: int = 10_000, seed: int = 47) -> dict[str, float | int | str]:
    ids = np.asarray(cluster_ids).reshape(-1)
    value_arr = np.asarray(values, dtype=float).reshape(-1)
    if ids.shape != value_arr.shape:
        raise ValueError("cluster_ids and values must have the same shape")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    unique = np.unique(ids)
    if unique.size == 0:
        raise ValueError("at least one cluster is required")
    generator = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for bootstrap_index in range(n_boot):
        sampled = generator.choice(unique, size=unique.size, replace=True)
        sample = np.concatenate([value_arr[ids == cluster] for cluster in sampled])
        estimates[bootstrap_index] = float(statistic(sample))
    finite = estimates[np.isfinite(estimates)]
    low, high = np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan)
    return {"unit": "event_cluster", "n": int(unique.size), "estimate": float(statistic(value_arr)), "ci_low": float(low), "ci_high": float(high)}


def unique_artifacts_by_sha256(paths: Sequence[Path]) -> tuple[list[Path], dict[Path, Path]]:
    unique: list[Path] = []
    first_by_hash: dict[str, Path] = {}
    duplicates: dict[Path, Path] = {}
    for path_like in paths:
        path = Path(path_like)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in first_by_hash:
            duplicates[path] = first_by_hash[digest]
        else:
            first_by_hash[digest] = path
            unique.append(path)
    return unique, duplicates


def event_cluster_bootstrap_metrics(labels: np.ndarray, predictions: np.ndarray, n_boot: int = 2_000, seed: int = 47) -> dict[str, object]:
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    pred_arr = np.asarray(predictions).astype(bool).reshape(-1)
    if label_arr.shape != pred_arr.shape:
        raise ValueError("labels and predictions must have the same shape")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    anomaly_events = events_from_binary(label_arr)
    if not anomaly_events:
        raise ValueError("at least one anomaly event is required")
    event_lengths = [stop - start + 1 for start, stop in anomaly_events]
    normal_block_length = max(1, int(np.median(event_lengths)))
    normal_blocks = _normal_blocks(label_arr, normal_block_length)
    if not normal_blocks:
        raise ValueError("at least one normal block is required")
    # A bootstrap replicate is a concatenation of sampled event and normal-block
    # clusters.  Confusion counts and event delays are additive over those
    # clusters, so aggregate their sufficient values instead of rebuilding a
    # full time series for every replicate.
    event_stats = np.asarray([_event_statistics(pred_arr, start, stop) for start, stop in anomaly_events], dtype=float)
    normal_fp = np.asarray([pred_arr[start:stop].sum() for start, stop in normal_blocks], dtype=float)
    generator = np.random.default_rng(seed)
    sampled_events = generator.integers(0, len(anomaly_events), size=(n_boot, len(anomaly_events)))
    sampled_normal = generator.integers(0, len(normal_blocks), size=(n_boot, len(normal_blocks)))
    event_sum = event_stats[sampled_events].sum(axis=1)
    normal_fp_sum = normal_fp[sampled_normal].sum(axis=1)
    tp, fp_event, fn, hits, delay_sum = event_sum.T
    fp = fp_event + normal_fp_sum
    precision_den = tp + fp
    recall_den = tp + fn
    precision = np.divide(tp, precision_den, out=np.zeros_like(tp), where=precision_den != 0)
    recall = np.divide(tp, recall_den, out=np.zeros_like(tp), where=recall_den != 0)
    f1_samples = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    recall_samples = hits / len(anomaly_events)
    mttd_samples = np.divide(delay_sum, hits, out=np.full(n_boot, np.nan), where=hits != 0)
    samples = {"raw_f1": f1_samples, "event_recall": recall_samples, "mttd": mttd_samples}
    point = {"raw_f1": float(f1(pred_arr, label_arr)), "event_recall": _event_recall(pred_arr, label_arr), "mttd": float(detection_delay(pred_arr, label_arr))}
    metrics: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        low, high = np.quantile(finite, [0.025, 0.975]) if finite.size else (np.nan, np.nan)
        metrics[name] = {"estimate": point[name], "ci_low": float(low), "ci_high": float(high)}
    return {"unit": "anomaly_event_and_normal_block", "anomaly_events": len(anomaly_events), "normal_blocks": len(normal_blocks), "metrics": metrics}


def _event_statistics(predictions: np.ndarray, start: int, stop: int) -> tuple[float, float, float, float, float]:
    """Return TP, FP, FN, hit indicator, and delay sum for one label event."""
    segment = np.asarray(predictions[start : stop + 1]).astype(bool)
    tp = float(segment.sum())
    fp = 0.0
    fn = float(segment.size - segment.sum())
    hits = float(segment.any())
    delay = float(np.flatnonzero(segment)[0]) if hits else 0.0
    return tp, fp, fn, hits, delay


def _normal_blocks(labels: np.ndarray, block_length: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for start, stop in events_from_binary(~labels):
        cursor, exclusive_stop = int(start), int(stop) + 1
        while cursor + block_length <= exclusive_stop:
            blocks.append((cursor, cursor + block_length))
            cursor += block_length
    return blocks


def _event_recall(predictions: np.ndarray, labels: np.ndarray) -> float:
    events = events_from_binary(labels)
    return float(np.mean([predictions[start : stop + 1].any() for start, stop in events])) if events else 0.0
