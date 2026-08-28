from __future__ import annotations

import numpy as np

from hbpc.adaptation_causal import (
    adaptation_features,
    compute_rank_biserial,
    fit_tau,
)


def test_fit_tau_recovers_exponential_decay_scale() -> None:
    tau_true = 4.0
    t = np.arange(20, dtype=float)
    values = 9.0 * np.exp(-t / tau_true) + 1.0

    result = fit_tau(values, max_steps=20)

    assert abs(result["tau"] - tau_true) < 1.0
    assert result["r_squared"] > 0.9


def test_fit_tau_marks_non_decaying_segments_as_infinite() -> None:
    values = np.array([1.0, 1.5, 2.0, 2.2, 2.4])

    result = fit_tau(values)

    assert np.isinf(result["tau"])
    assert result["r_squared"] == 0.0


def test_adaptation_features_compare_anomaly_and_normal_high_segments() -> None:
    errors = np.zeros((80, 2), dtype=float)
    labels = np.zeros(80, dtype=bool)
    labels[40:55] = True

    errors[10:15, 0] = [8.0, 3.0, 1.5, 1.0, 0.8]
    errors[40:55, :] = np.column_stack(
        [
            np.linspace(8.0, 4.0, 15),
            np.linspace(6.0, 3.0, 15),
        ]
    )
    scores = np.linalg.norm(errors, axis=1)

    features = adaptation_features(errors, labels, scores, max_steps=10, high_fraction=0.1)

    assert features["tau_anomaly_n"] >= 1
    assert features["tau_normal_n"] >= 1
    assert features["tau_ratio"] > 1.0


def test_compute_rank_biserial_uses_high_normal_and_high_anomaly_scores() -> None:
    scores = np.full(100, 0.1, dtype=float)
    labels = np.zeros(100, dtype=bool)
    labels[60:90] = True
    scores[10] = 10.0
    scores[11:14] = 0.1
    scores[60:80] = 8.0

    effect = compute_rank_biserial(scores, labels, k=3, high_fraction=0.01)

    assert effect > 0.5
