import numpy as np

from hbpc.ceres import (
    assign_bins,
    build_tail_envelopes,
    ceres_envelope_score,
    ceres_lite_score,
    normalize_scores,
)


def test_normalize_scores_uses_median_iqr_scale():
    z, scale = normalize_scores(np.array([1.0, 2.0, 3.0]))

    assert scale > 0
    assert np.all(np.isfinite(z))
    assert np.all(z >= 0)


def test_assign_bins_returns_effective_bin_indices():
    values = np.array([0.1, 0.6, 0.96])
    edges = np.array([0.0, 0.5, 0.9, 1.0])

    bins = assign_bins(values, edges)

    assert bins.tolist() == [0, 1, 2]


def test_build_tail_envelopes_merges_sparse_bins():
    z = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

    env, diagnostics = build_tail_envelopes(
        z,
        k=2,
        bin_probs=[0.0, 0.5, 0.9, 1.0],
        q_upper=0.9,
        min_bin_count=3,
    )

    assert env["tail_quantiles"].shape[0] == len(diagnostics)
    assert env["step_quantiles"].shape[1] == 2
    assert "sample_count" in diagnostics[0]
    assert any(row["merged_from"] for row in diagnostics)


def test_ceres_lite_score_is_positive_only_for_excess_tail():
    z = np.array([1, 1, 10, 10, 10, 1], dtype=float)
    env, _ = build_tail_envelopes(
        z[:4],
        k=1,
        bin_probs=[0.0, 1.0],
        q_upper=0.5,
        min_bin_count=1,
    )

    scores = ceres_lite_score(z, env, k=1)

    assert np.nanmax(scores) >= 0
    assert scores[0] == 0


def test_ceres_envelope_score_uses_stepwise_excess():
    z = np.array([1, 1, 6, 8, 7, 1], dtype=float)
    env, _ = build_tail_envelopes(
        z[:4],
        k=2,
        bin_probs=[0.0, 1.0],
        q_upper=0.5,
        min_bin_count=1,
    )

    scores = ceres_envelope_score(z, env, k=2)

    assert np.nanmax(scores) >= 0
    assert np.isnan(scores[-1])
