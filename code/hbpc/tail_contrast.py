from __future__ import annotations

import numpy as np
import pandas as pd

from hbpc.metrics import events_from_binary


ANCHOR_COLUMNS = ["class_label", "event_id", "index", "peak_score", "log_peak_score", "tail_mean"]


def normalize_scores(scores: np.ndarray, training_normal_scores: np.ndarray) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    reference = np.asarray(training_normal_scores, dtype=float).reshape(-1)
    reference = reference[np.isfinite(reference)]
    if reference.size < 2:
        raise ValueError("training-normal reference must contain at least two finite scores")
    q25, median, q75 = np.quantile(reference, [0.25, 0.5, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("training-normal interquartile range must be positive")
    return (scores_arr - float(median)) / scale


def extract_event_anchors(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
    normal_threshold: float,
    normalized_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be positive")
    raw = np.asarray(scores, dtype=float).reshape(-1)
    label_arr = np.asarray(labels).astype(bool).reshape(-1)
    normalized = raw if normalized_scores is None else np.asarray(normalized_scores, dtype=float).reshape(-1)
    if raw.shape != label_arr.shape or raw.shape != normalized.shape:
        raise ValueError("scores, normalized scores, and labels must have the same shape")
    if not np.isfinite(normal_threshold):
        raise ValueError("normal_threshold must be finite")

    rows: list[dict[str, object]] = []
    for event_id, (start, stop) in enumerate(events_from_binary(label_arr)):
        valid_stop = min(int(stop), raw.size - k - 1)
        if valid_stop < int(start):
            continue
        candidates = np.arange(int(start), valid_stop + 1)
        finite = candidates[np.isfinite(raw[candidates])]
        if finite.size:
            anchor = int(finite[np.argmax(raw[finite])])
            rows.append(_anchor_row(raw, normalized, anchor, k, 1, f"A{event_id}"))

    candidates: list[int] = []
    for index in range(0, raw.size - k):
        if (
            not np.isfinite(raw[index])
            or not np.isfinite(normalized[index])
            or normalized[index] < normal_threshold
        ):
            continue
        if label_arr[index : index + k + 1].any():
            continue
        left, right = max(0, index - k), min(raw.size, index + k + 1)
        neighbors = raw[left:right]
        if np.isfinite(neighbors).any() and raw[index] >= np.nanmax(neighbors):
            candidates.append(index)

    retained: list[int] = []
    for index in sorted(candidates, key=lambda value: (-raw[value], value)):
        if all(abs(index - prior) >= 2 * k + 1 for prior in retained):
            retained.append(index)
    for event_id, anchor in enumerate(sorted(retained)):
        rows.append(_anchor_row(raw, normalized, anchor, k, 0, f"N{event_id}"))
    return pd.DataFrame(rows, columns=ANCHOR_COLUMNS).sort_values(["class_label", "index"], ignore_index=True)


def match_peak_anchors(anchors: pd.DataFrame, caliper: float = 0.2) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if caliper <= 0.0:
        raise ValueError("caliper must be positive")
    missing = set(ANCHOR_COLUMNS) - set(anchors.columns)
    if missing:
        raise ValueError(f"anchor table is missing columns: {sorted(missing)}")
    normal = anchors[anchors["class_label"] == 0].copy()
    anomaly = anchors[anchors["class_label"] == 1].copy()
    pooled = anchors["log_peak_score"].to_numpy(dtype=float)
    finite = pooled[np.isfinite(pooled)]
    scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    columns = ["anomaly_index", "normal_index", "anomaly_peak", "normal_peak", "peak_distance", "anomaly_tail_mean", "normal_tail_mean", "tail_difference"]
    if normal.empty or anomaly.empty or scale <= 0.0:
        return pd.DataFrame(columns=columns), {"matched_pairs": 0, "standardized_mean_difference": float("nan")}
    center = float(np.mean(finite))
    normal["peak_z"] = (normal["log_peak_score"] - center) / scale
    anomaly["peak_z"] = (anomaly["log_peak_score"] - center) / scale
    available = set(normal.index.tolist())
    pairs: list[dict[str, float | int]] = []
    for _, anomaly_row in anomaly.sort_values(["peak_z", "index"], ascending=[False, True]).iterrows():
        if not available:
            break
        normal_rows = normal.loc[sorted(available)].copy()
        normal_rows["distance"] = np.abs(normal_rows["peak_z"] - float(anomaly_row["peak_z"]))
        normal_row = normal_rows.sort_values(["distance", "index"]).iloc[0]
        distance = float(normal_row["distance"])
        if distance > caliper:
            continue
        available.remove(normal_row.name)
        pairs.append({
            "anomaly_index": int(anomaly_row["index"]), "normal_index": int(normal_row["index"]),
            "anomaly_peak": float(anomaly_row["peak_z"]), "normal_peak": float(normal_row["peak_z"]),
            "peak_distance": distance, "anomaly_tail_mean": float(anomaly_row["tail_mean"]),
            "normal_tail_mean": float(normal_row["tail_mean"]),
            "tail_difference": float(anomaly_row["tail_mean"] - normal_row["tail_mean"]),
        })
    matched = pd.DataFrame(pairs, columns=columns)
    smd = float(matched["anomaly_peak"].mean() - matched["normal_peak"].mean()) if not matched.empty else float("nan")
    return matched, {"matched_pairs": int(len(matched)), "standardized_mean_difference": smd}


def paired_tail_contrast(matched: pd.DataFrame) -> dict[str, float | int]:
    if "tail_difference" not in matched or matched.empty:
        raise ValueError("matched event pairs are required")
    differences = matched["tail_difference"].to_numpy(dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("matched event pairs contain no finite tail differences")
    return {"matched_pairs": int(differences.size), "tail_contrast": float(np.mean(differences))}


def _anchor_row(raw: np.ndarray, normalized: np.ndarray, index: int, k: int, class_label: int, event_id: str) -> dict[str, object]:
    tail = normalized[index + 1 : index + k + 1]
    finite_tail = tail[np.isfinite(tail)]
    return {
        "class_label": int(class_label), "event_id": event_id, "index": int(index),
        "peak_score": float(raw[index]), "log_peak_score": float(np.log1p(max(raw[index], 0.0))),
        "tail_mean": float(np.mean(finite_tail)) if finite_tail.size == k else float("nan"),
    }
