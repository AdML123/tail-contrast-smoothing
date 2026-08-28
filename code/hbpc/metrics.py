import numpy as np


def events_from_binary(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels).astype(bool)
    events: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(labels):
        if value and start is None:
            start = i
        if (not value) and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(labels) - 1))
    return events


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=np.int64).copy()
    for start, stop in events_from_binary(labels):
        if adjusted[start : stop + 1].any():
            adjusted[start : stop + 1] = 1
    return adjusted


def f1(pred: np.ndarray, labels: np.ndarray) -> float:
    pred = np.asarray(pred).astype(bool)
    labels = np.asarray(labels).astype(bool)
    tp = np.logical_and(pred, labels).sum()
    fp = np.logical_and(pred, ~labels).sum()
    fn = np.logical_and(~pred, labels).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def detection_delay(pred: np.ndarray, labels: np.ndarray) -> float:
    delays: list[int] = []
    for start, stop in events_from_binary(labels):
        hits = np.flatnonzero(np.asarray(pred)[start : stop + 1] > 0)
        if len(hits):
            delays.append(int(hits[0]))
    return float(np.mean(delays)) if delays else float("nan")


def affiliation_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    pred_events = events_from_binary(pred)
    gt_events = events_from_binary(labels)
    if not pred_events or not gt_events:
        return 0.0
    try:
        from affiliation.metrics import pr_from_events

        result = pr_from_events(pred_events, gt_events, Trange=(0, len(labels) - 1))
        precision = result["precision"]
        recall = result["recall"]
        return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    except Exception:
        return 0.0
