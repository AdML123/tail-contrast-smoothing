from __future__ import annotations

import numpy as np

from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.score_benchmark import (
    backward_average_score,
    benchmark_score_vector,
    confirmation_window_score,
    ewma_score,
    forward_average_score,
    top_n_predictions,
)


def test_future_tail_at_anchor_equals_causal_confirmation_at_issuance():
    scores = np.array([1.0, 9.0, 7.0, 5.0, 3.0, 2.0])
    k = 2
    future = forward_average_score(scores, k=k)
    causal = confirmation_window_score(scores, k=k)
    assert np.allclose(future[:-k], causal[k:], equal_nan=True)


def test_forward_average_score_uses_future_window_and_nan_tail():
    scores = np.array([1.0, 9.0, 7.0, 5.0, 3.0])
    out = forward_average_score(scores, k=2)
    assert np.allclose(out[:3], [8.0, 6.0, 4.0])
    assert np.isnan(out[3])
    assert np.isnan(out[4])


def test_backward_average_score_uses_causal_window_and_nan_head():
    scores = np.array([1.0, 9.0, 7.0, 5.0, 3.0])
    out = backward_average_score(scores, k=2)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [5.0, 8.0, 6.0, 4.0])


def test_ewma_score_matches_recursive_definition():
    scores = np.array([1.0, 3.0, 5.0])
    out = ewma_score(scores, alpha=0.5)
    assert np.allclose(out, [1.0, 2.0, 3.5])


def test_top_n_predictions_can_shift_forward_alarms():
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    pred = top_n_predictions(scores, top_n=2, delay=1, length=5)
    assert pred.tolist() == [0, 0, 1, 0, 1]


def test_benchmark_exposes_one_confirmation_window_method():
    scores = np.array([1.0, 9.0, 7.0, 6.0, 1.0, 1.0])
    labels = np.array([0, 1, 1, 1, 0, 0])
    frame = benchmark_score_vector(
        scores,
        labels,
        dataset="toy",
        predictor="p",
        seed=0,
        top_ns=(1,),
        windows=(2,),
        ewma_alphas=(0.5,),
    )
    methods = set(frame["postprocess"])
    assert {"raw", "ewma", "confirmation_mean"}.issubset(methods)
    assert "forward_avg" not in methods
    assert "backward_avg" not in methods
    confirmation = frame[frame["postprocess"] == "confirmation_mean"].iloc[0]
    assert confirmation["k"] == 2
    assert confirmation["delay"] == 0
    assert 0.0 <= confirmation["raw_f1"] <= 1.0


def test_benchmark_score_vector_reports_event_precision_f1_and_delayed_controls():
    scores = np.array([1.0, 9.0, 7.0, 6.0, 1.0, 8.0, 7.0, 1.0])
    labels = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    frame = benchmark_score_vector(
        scores,
        labels,
        dataset="toy",
        predictor="p",
        seed=0,
        top_ns=(2,),
        windows=(2,),
        ewma_alphas=(0.5,),
        include_delayed_controls=True,
    )

    assert {"event_precision", "event_f1"}.issubset(frame.columns)
    assert frame["event_precision"].between(0.0, 1.0).all()
    assert frame["event_f1"].between(0.0, 1.0).all()
    assert {"raw_delayed", "ewma_delayed"}.issubset(set(frame["postprocess"]))
    delayed = frame[frame["postprocess"] == "raw_delayed"].iloc[0]
    assert delayed["k"] == 2
    assert delayed["delay"] == 2


def test_forward_and_backward_average_ignore_non_finite_values():
    scores = np.array([1.0, np.nan, 5.0, 7.0, 9.0])
    forward = forward_average_score(scores, k=2)
    backward = backward_average_score(scores, k=2)
    assert np.allclose(forward[:3], [5.0, 6.0, 8.0], equal_nan=True)
    assert np.isnan(forward[3])
    assert np.isnan(forward[4])
    assert np.isnan(backward[0])
    assert np.allclose(backward[1:], [1.0, 5.0, 6.0, 8.0], equal_nan=True)


def test_benchmark_score_vector_matches_dense_metric_reference():
    rng = np.random.default_rng(13)
    scores = rng.normal(size=64)
    scores[[3, 17, 41]] = [5.0, 4.5, 6.0]
    labels = np.zeros(64, dtype=int)
    labels[10:14] = 1
    labels[40:45] = 1

    frame = benchmark_score_vector(
        scores,
        labels,
        dataset="toy",
        predictor="p",
        seed=0,
        top_ns=(6,),
        windows=(3,),
        ewma_alphas=(0.4,),
        include_delayed_controls=True,
    )

    for row in frame.to_dict("records"):
        if row["postprocess"] == "raw":
            score = scores
        elif row["postprocess"] == "raw_delayed":
            score = scores
        elif row["postprocess"] == "ewma":
            score = ewma_score(scores, alpha=float(row["alpha"]))
        elif row["postprocess"] == "ewma_delayed":
            score = ewma_score(scores, alpha=float(row["alpha"]))
        elif row["postprocess"] in {"confirmation_mean", "backward_avg", "backward_avg_delayed"}:
            score = confirmation_window_score(scores, k=int(row["k"]))
        else:
            raise AssertionError(row["postprocess"])
        pred = top_n_predictions(score, top_n=int(row["top_n"]), delay=int(row["delay"]), length=len(labels))
        adjusted = point_adjust(pred, labels)
        pred_events = events_from_binary(pred)
        label_events = events_from_binary(labels)
        if label_events:
            event_recall = sum(bool(pred[start : stop + 1].any()) for start, stop in label_events) / len(label_events)
        else:
            event_recall = 0.0
        if pred_events:
            event_precision = sum(bool(labels[start : stop + 1].any()) for start, stop in pred_events) / len(pred_events)
        else:
            event_precision = 0.0
        event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if event_precision + event_recall else 0.0

        assert row["raw_f1"] == f1(pred, labels)
        assert row["pa_f1"] == f1(adjusted, labels)
        assert row["event_recall"] == event_recall
        assert row["event_precision"] == event_precision
        assert row["event_f1"] == event_f1
        actual_delay = row["mttd"]
        expected_delay = detection_delay(pred, labels)
        assert actual_delay == expected_delay or (np.isnan(actual_delay) and np.isnan(expected_delay))
