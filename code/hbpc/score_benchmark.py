from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust


@dataclass(frozen=True)
class ScoreRun:
    dataset: str
    predictor: str
    seed: int
    scores: np.ndarray
    labels: np.ndarray
    segment_lengths: tuple[int, ...] | None = None


def forward_average_score(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    out = np.full(arr.size, np.nan, dtype=float)
    if arr.size <= k:
        return out
    values = np.where(np.isfinite(arr), arr, 0.0)
    counts = np.isfinite(arr).astype(float)
    value_cumsum = np.concatenate([[0.0], np.cumsum(values)])
    count_cumsum = np.concatenate([[0.0], np.cumsum(counts)])
    starts = np.arange(1, arr.size - k + 1)
    stops = starts + k
    totals = value_cumsum[stops] - value_cumsum[starts]
    ns = count_cumsum[stops] - count_cumsum[starts]
    valid = ns > 0
    out[: arr.size - k][valid] = totals[valid] / ns[valid]
    return out


def backward_average_score(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    out = np.full(arr.size, np.nan, dtype=float)
    if arr.size < k:
        return out
    values = np.where(np.isfinite(arr), arr, 0.0)
    counts = np.isfinite(arr).astype(float)
    value_cumsum = np.concatenate([[0.0], np.cumsum(values)])
    count_cumsum = np.concatenate([[0.0], np.cumsum(counts)])
    starts = np.arange(0, arr.size - k + 1)
    stops = starts + k
    totals = value_cumsum[stops] - value_cumsum[starts]
    ns = count_cumsum[stops] - count_cumsum[starts]
    valid = ns > 0
    out[k - 1 :][valid] = totals[valid] / ns[valid]
    return out


def confirmation_window_score(scores: np.ndarray, k: int) -> np.ndarray:
    """Mean of the latest ``k`` scores, available causally at the current index."""
    return backward_average_score(scores, k=k)


def ewma_score(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    out = np.zeros(arr.size, dtype=float)
    out[0] = arr[0] if np.isfinite(arr[0]) else 0.0
    for idx in range(1, arr.size):
        value = arr[idx] if np.isfinite(arr[idx]) else 0.0
        out[idx] = alpha * value + (1.0 - alpha) * out[idx - 1]
    return out


def top_n_predictions(scores: np.ndarray, top_n: int, delay: int = 0, length: int | None = None) -> np.ndarray:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    output_length = arr.size if length is None else int(length)
    pred = np.zeros(output_length, dtype=np.int64)
    candidates = np.flatnonzero(np.isfinite(arr) & (arr > 0.0))
    if not candidates.size:
        return pred
    take = min(top_n, candidates.size)
    values = arr[candidates]
    if take == candidates.size:
        selected_base = candidates[np.lexsort((candidates, -values))]
    else:
        rough = np.argpartition(-values, take - 1)[:take]
        rough_indices = candidates[rough]
        rough_values = arr[rough_indices]
        selected_base = rough_indices[np.lexsort((rough_indices, -rough_values))]
    selected = selected_base.astype(int) + delay
    selected = selected[(selected >= 0) & (selected < output_length)]
    pred[selected] = 1
    return pred


def benchmark_score_vector(
    scores: np.ndarray,
    labels: np.ndarray,
    dataset: str,
    predictor: str,
    seed: int,
    top_ns: Sequence[int] = (100, 300, 500),
    windows: Sequence[int] = (1, 2, 3, 5, 10, 20),
    ewma_alphas: Sequence[float] = (0.3,),
    include_delayed_controls: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    arr = np.asarray(scores, dtype=float).reshape(-1)
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    if arr.shape != label_arr.shape:
        raise ValueError("scores and labels must have the same shape")

    score_specs: list[tuple[str, np.ndarray, int, int, float | None]] = [("raw", arr, 0, 0, None)]
    ewma_cache: dict[float, np.ndarray] = {}
    for alpha in ewma_alphas:
        ewma = ewma_score(arr, alpha=float(alpha))
        ewma_cache[float(alpha)] = ewma
        score_specs.append(("ewma", ewma, 0, 0, float(alpha)))

    for k in windows:
        k_int = int(k)
        confirmation = confirmation_window_score(arr, k=k_int)
        score_specs.append(("confirmation_mean", confirmation, k_int, 0, None))
        if include_delayed_controls:
            score_specs.append(("raw_delayed", arr, k_int, k_int, None))
            for alpha, ewma in ewma_cache.items():
                score_specs.append(("ewma_delayed", ewma, k_int, k_int, float(alpha)))

    for top_n in top_ns:
        for postprocess, score, k, delay, alpha in score_specs:
            rows.append(
                _metric_row(
                    dataset,
                    predictor,
                    seed,
                    postprocess,
                    score,
                    label_arr,
                    int(top_n),
                    k=int(k),
                    delay=int(delay),
                    alpha=alpha,
                )
            )
    return pd.DataFrame(rows)

def load_npz_score_run(path: Path | str, dataset: str, predictor: str, seed: int) -> ScoreRun:
    data = np.load(Path(path))
    if "scores" not in data or "labels" not in data:
        raise ValueError(f"{path} must contain scores and labels arrays")
    return ScoreRun(
        dataset=dataset,
        predictor=predictor,
        seed=int(seed),
        scores=np.asarray(data["scores"], dtype=float).reshape(-1),
        labels=np.asarray(data["labels"]).astype(bool).reshape(-1),
    )


def benchmark_runs(runs: Sequence[ScoreRun], output_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        benchmark_score_vector(
            run.scores,
            run.labels,
            dataset=run.dataset,
            predictor=run.predictor,
            seed=run.seed,
        )
        for run in runs
    ]
    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    best = best_by_predictor(all_rows)
    output_path = Path(output_dir)
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output_path / "metrics" / "postprocess_all_rows.csv", index=False)
    best.to_csv(output_path / "tables" / "postprocess_best_by_predictor.csv", index=False)
    return all_rows, best


def best_by_predictor(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    rows: list[pd.Series] = []
    for (dataset, predictor, postprocess), subset in frame.groupby(["dataset", "predictor", "postprocess"]):
        idx = subset.sort_values(
            ["raw_f1", "pa_f1", "event_recall", "mttd"],
            ascending=[False, False, False, True],
        ).index[0]
        row = subset.loc[idx].copy()
        row["dataset"] = dataset
        row["predictor"] = predictor
        row["postprocess"] = postprocess
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _metric_row(
    dataset: str,
    predictor: str,
    seed: int,
    postprocess: str,
    score: np.ndarray,
    labels: np.ndarray,
    top_n: int,
    k: int,
    delay: int,
    alpha: float | None = None,
) -> dict[str, object]:
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    alarm_idx = _top_n_alarm_indices(score, top_n=top_n, delay=delay, length=label_arr.size)
    label_events = events_from_binary(label_arr)
    pred_events = _events_from_sorted_indices(alarm_idx)
    label_points = int(label_arr.sum())
    tp = int(label_arr[alarm_idx].sum()) if alarm_idx.size else 0
    fp = int(alarm_idx.size - tp)
    fn = int(label_points - tp)
    raw_f1 = _f1_from_counts(tp, fp, fn)

    hit_events: list[tuple[int, int]] = []
    delays: list[int] = []
    for start, stop in label_events:
        pos = _first_index_in_interval(alarm_idx, start, stop)
        if pos is not None:
            hit_events.append((start, stop))
            delays.append(int(pos - start))
    adjusted_tp = int(sum(stop - start + 1 for start, stop in hit_events))
    adjusted_fp = fp
    adjusted_fn = int(label_points - adjusted_tp)
    event_recall = float(len(hit_events) / len(label_events)) if label_events else 0.0
    event_precision = _event_precision_from_events(pred_events, label_arr)
    event_f1 = float(2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) else 0.0
    return {
        "dataset": dataset,
        "predictor": predictor,
        "seed": int(seed),
        "postprocess": postprocess,
        "top_n": int(top_n),
        "k": int(k),
        "delay": int(delay),
        "alpha": np.nan if alpha is None else float(alpha),
        "raw_f1": raw_f1,
        "pa_f1": _f1_from_counts(adjusted_tp, adjusted_fp, adjusted_fn),
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "mttd": float(np.mean(delays)) if delays else float("nan"),
        "predicted_points": int(alarm_idx.size),
        "predicted_events": int(len(pred_events)),
        "label_points": label_points,
        "label_events": int(len(label_events)),
    }


def _top_n_alarm_indices(scores: np.ndarray, top_n: int, delay: int, length: int) -> np.ndarray:
    pred = top_n_predictions(scores, top_n=top_n, delay=delay, length=length)
    return np.flatnonzero(pred > 0)


def _events_from_sorted_indices(indices: np.ndarray) -> list[tuple[int, int]]:
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size == 0:
        return []
    events: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        current = int(value)
        if current == prev + 1:
            prev = current
        else:
            events.append((start, prev))
            start = current
            prev = current
    events.append((start, prev))
    return events


def _first_index_in_interval(indices: np.ndarray, start: int, stop: int) -> int | None:
    left = int(np.searchsorted(indices, start, side="left"))
    if left < indices.size and int(indices[left]) <= int(stop):
        return int(indices[left])
    return None


def _event_precision_from_events(pred_events: list[tuple[int, int]], labels: np.ndarray) -> float:
    if not pred_events:
        return 0.0
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    hits = sum(bool(label_arr[start : stop + 1].any()) for start, stop in pred_events)
    return float(hits / len(pred_events))


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

def _event_recall(pred: np.ndarray, labels: np.ndarray) -> float:
    events = events_from_binary(labels)
    if not events:
        return 0.0
    pred_bool = np.asarray(pred).astype(bool).reshape(-1)
    hits = sum(bool(pred_bool[start : stop + 1].any()) for start, stop in events)
    return float(hits / len(events))


def _event_precision(pred: np.ndarray, labels: np.ndarray) -> float:
    events = events_from_binary(pred)
    if not events:
        return 0.0
    label_bool = np.asarray(labels).astype(bool).reshape(-1)
    hits = sum(bool(label_bool[start : stop + 1].any()) for start, stop in events)
    return float(hits / len(events))


def _event_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    precision = _event_precision(pred, labels)
    recall = _event_recall(pred, labels)
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
