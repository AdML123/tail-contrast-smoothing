import numpy as np

from hbpc.mars import tail_score
from hbpc.scar import build_scar_calibration, scar_score


def test_scar_score_is_never_below_tail_score():
    residuals = np.array([1, 1, 9, 7, 6, 5, 1, 1], dtype=float)
    calibration, _ = build_scar_calibration(
        residuals,
        k=3,
        n_bins=3,
        q_upper=0.5,
        n_ref=2,
        min_bin_count=1,
    )

    scores = scar_score(residuals, calibration, k=3)
    tail = tail_score(residuals, k=3)
    valid = np.isfinite(scores) & np.isfinite(tail)

    assert valid.any()
    assert np.all(scores[valid] >= tail[valid] - 1e-12)


def test_sparse_bin_falls_back_to_tail_score():
    residuals = np.array([1, 1, 9, 7, 6, 5, 1, 1], dtype=float)
    calibration, diagnostics = build_scar_calibration(
        residuals,
        k=3,
        n_bins=4,
        q_upper=0.95,
        n_ref=200,
        min_bin_count=100,
    )

    scores = scar_score(residuals, calibration, k=3)
    tail = tail_score(residuals, k=3)
    valid = np.isfinite(scores) & np.isfinite(tail)

    np.testing.assert_allclose(scores[valid], tail[valid])
    assert all(row["w"] == 0.0 for row in diagnostics)


def test_bin_diagnostics_include_label_counts():
    residuals = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)

    _, diagnostics = build_scar_calibration(
        residuals,
        k=2,
        n_bins=2,
        q_upper=0.9,
        n_ref=3,
        min_bin_count=1,
        labels=labels,
    )

    assert {"bin_id", "bin_low", "bin_high", "n_b", "Q_b", "w", "label_normal_count", "label_anomaly_count"}.issubset(
        diagnostics[0].keys()
    )
    assert sum(row["label_normal_count"] for row in diagnostics) > 0
    assert sum(row["label_anomaly_count"] for row in diagnostics) > 0
