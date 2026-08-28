from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimeSeriesLibraryArrays:
    train: Path
    test: Path
    test_label: Path


def resolve_time_series_library_arrays(data_root: Path | str, dataset: str) -> TimeSeriesLibraryArrays:
    root = Path(data_root)
    layouts = [
        (
            root / f"{dataset}_train.npy",
            root / f"{dataset}_test.npy",
            root / f"{dataset}_test_label.npy",
        ),
        (
            root / dataset / f"{dataset}_train.npy",
            root / dataset / f"{dataset}_test.npy",
            root / dataset / f"{dataset}_test_label.npy",
        ),
    ]
    for train, test, test_label in layouts:
        if train.exists() and test.exists() and test_label.exists():
            return TimeSeriesLibraryArrays(train=train, test=test, test_label=test_label)
    expected = " or ".join(str(paths[0].parent) for paths in layouts)
    raise FileNotFoundError(
        f"Could not find {dataset} train/test/test_label arrays under {root}. "
        f"Expected flat or Hugging Face dataset layout in: {expected}"
    )


def make_sliding_windows(data: np.ndarray, win_size: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if win_size <= 0:
        raise ValueError("win_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array of shape [time, channels]")
    if arr.shape[0] < win_size:
        return np.empty((0, win_size, arr.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    starts = np.arange(0, arr.shape[0] - win_size + 1, step, dtype=np.int64)
    windows = np.stack([arr[start : start + win_size] for start in starts]).astype(np.float32)
    return windows, starts


def aggregate_window_scores_to_points(window_scores: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    scores = np.asarray(window_scores, dtype=float)
    start_arr = np.asarray(starts, dtype=np.int64).reshape(-1)
    if scores.ndim != 2:
        raise ValueError("window_scores must be a 2D array of shape [windows, win_size]")
    if scores.shape[0] != start_arr.size:
        raise ValueError("window_scores and starts must contain the same number of windows")
    total = np.zeros(int(length), dtype=float)
    counts = np.zeros(int(length), dtype=float)
    for row, start in zip(scores, start_arr):
        end = min(int(start) + row.size, int(length))
        usable = max(0, end - int(start))
        if usable:
            total[int(start) : end] += row[:usable]
            counts[int(start) : end] += 1.0
    out = np.full(int(length), np.nan, dtype=float)
    mask = counts > 0
    out[mask] = total[mask] / counts[mask]
    return out


def save_score_npz(output_path: Path | str, scores: np.ndarray, labels: np.ndarray) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    if score_arr.shape != label_arr.shape:
        raise ValueError("scores and labels must have the same shape")
    np.savez_compressed(path, scores=score_arr, labels=label_arr)


def ensure_same_length(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    label_arr = np.asarray(labels).astype(np.int64).reshape(-1)
    finite = np.flatnonzero(np.isfinite(score_arr))
    if finite.size:
        start = int(finite[0])
        stop = int(finite[-1]) + 1
        score_arr = score_arr[start:stop]
        label_arr = label_arr[start:stop]
    min_len = min(score_arr.size, label_arr.size)
    return score_arr[:min_len], label_arr[:min_len]
