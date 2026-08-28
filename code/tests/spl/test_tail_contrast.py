from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hbpc.tail_contrast import extract_event_anchors, match_peak_anchors, normalize_scores


def test_extract_event_anchors_uses_one_peak_per_anomaly_event():
    scores = np.array([0, 2, 5, 4, 0, 1, 7, 6, 0, 3, 0, 0], dtype=float)
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    anchors = extract_event_anchors(scores, labels, k=2, normal_threshold=1.0)
    anomaly = anchors[anchors["class_label"] == 1]
    assert anomaly["index"].tolist() == [2, 6]


def test_normal_anchor_tails_do_not_overlap_each_other_or_anomalies():
    scores = np.array([4, 3, 2, 0, 5, 4, 3, 0, 9, 8, 0, 0], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    anchors = extract_event_anchors(scores, labels, k=2, normal_threshold=1.0)
    normal = anchors[anchors["class_label"] == 0]["index"].to_numpy()
    assert np.all(np.diff(normal) >= 5)
    assert all(not labels[index : index + 3].any() for index in normal)


def test_normal_anchor_threshold_is_compared_in_normalized_scale():
    scores = np.array([0.0, 10.0, 0.0, 0.0, 9.0, 0.0])
    labels = np.zeros(scores.size, dtype=bool)
    normalized = np.array([0.0, 1.0, 0.0, 0.0, 1.5, 0.0])
    anchors = extract_event_anchors(
        scores,
        labels,
        k=1,
        normal_threshold=2.0,
        normalized_scores=normalized,
    )
    assert anchors[anchors["class_label"] == 0].empty


def _make_anchor_frame(normal_peaks, anomaly_peaks):
    rows = []
    for class_label, peaks in ((0, normal_peaks), (1, anomaly_peaks)):
        for index, peak in enumerate(peaks):
            rows.append(
                {
                    "class_label": class_label,
                    "event_id": f"{class_label}-{index}",
                    "index": index,
                    "peak_score": float(peak),
                    "log_peak_score": float(np.log1p(peak)),
                    "tail_mean": float(peak / 10.0),
                }
            )
    return pd.DataFrame(rows)


def test_match_peak_anchors_is_one_to_one_and_respects_caliper():
    anchors = _make_anchor_frame([9.9, 20.0, 40.0], [10.0, 39.5])
    matched, balance = match_peak_anchors(anchors, caliper=0.2)
    assert len(matched) == 2
    assert matched["normal_index"].is_unique
    assert matched["anomaly_index"].is_unique
    assert matched["peak_distance"].max() <= 0.2
    assert abs(balance["standardized_mean_difference"]) < 0.1


def test_normalization_rejects_zero_iqr():
    with pytest.raises(ValueError, match="interquartile range"):
        normalize_scores(np.array([1.0, 2.0]), np.ones(10))


def test_matching_returns_empty_result_when_no_peak_is_within_caliper():
    anchors = _make_anchor_frame([1.0], [100.0])
    matched, balance = match_peak_anchors(anchors, caliper=0.01)
    assert matched.empty
    assert balance["matched_pairs"] == 0
