import numpy as np

from hbpc.rrp import (
    assign_rrp_groups,
    candidate_count,
    cusum_scores,
    delayed_top_n_alarms,
    ewma_scores,
    evaluate_phenomenon_gate,
    group_summary,
    peak_gated_tail_scores,
    peak_tail_scores,
    relaxation_features,
    rank_effect,
    tail_scores,
    top_n_alarms,
)


def test_relaxation_features_use_complete_future_windows():
    scores = np.array([2.0, 1.0, 1.0, 8.0, 4.0, 2.0])

    features = relaxation_features(scores, k=2)

    np.testing.assert_array_equal(features.indices, np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(features.peak, np.array([2.0, 1.0, 1.0, 8.0]))
    np.testing.assert_allclose(features.tail, np.array([1.0, 4.5, 6.0, 3.0]))
    np.testing.assert_allclose(
        features.relax,
        np.array([0.5, 4.5, 6.0, 0.375]),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        features.curves[0],
        np.array([1.0, 0.5, 0.5]),
        rtol=1e-6,
    )
    assert features.curves.shape == (4, 3)


def test_assign_rrp_groups_uses_separate_normal_and_anomaly_quantiles_and_event_lengths():
    scores = np.array(
        [
            1.0,
            2.0,
            3.0,
            10.0,
            4.0,
            22.0,
            12.0,
            13.0,
            14.0,
            6.0,
            20.0,
            21.0,
            7.0,
            8.0,
            9.0,
        ]
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    features = relaxation_features(scores, k=3)

    groups = assign_rrp_groups(
        features,
        labels,
        k=3,
        high_fraction=0.25,
        typical_quantiles=(0.4, 0.6),
    )

    assert groups[3] == "A"
    assert groups[5] == "B2"
    assert groups[11] == "B1"
    assert groups[9] == "C"
    assert groups[0] == "other"


def test_rank_effect_reports_directional_effect_size_and_mann_whitney_u():
    first = np.array([1.0, 2.0])
    second = np.array([2.0, 4.0])

    effect = rank_effect(first, second)

    assert effect.n_first == 2
    assert effect.n_second == 2
    assert effect.u == 3.5
    assert effect.common_language == 0.875
    assert effect.rank_biserial == 0.75


def test_group_summary_and_gate_focus_on_a_vs_persistent_anomaly():
    groups = np.array(["A", "A", "B2", "B2", "B2", "C"], dtype=object)
    relax = np.array([1.0, 1.2, 2.0, 2.4, 3.0, 0.8])
    peak = np.ones_like(relax)
    tail = relax.copy()

    summary = group_summary(groups, relax, peak, tail, k=5, seed=0)
    gate = evaluate_phenomenon_gate(
        effects_by_k={
            3: {"A_vs_B2": rank_effect(np.array([1.0] * 5), np.array([2.0] * 5))},
            5: {"A_vs_B2": rank_effect(np.array([1.0] * 5), np.array([2.0] * 5))},
            10: {"A_vs_B2": rank_effect(np.array([1.0] * 5), np.array([2.0] * 5))},
            20: {"A_vs_B2": rank_effect(np.array([1.0] * 5), np.array([1.1] * 5))},
        },
        min_group_size=5,
        min_rank_biserial=0.3,
        min_median_ratio=1.5,
        min_pass_horizons=3,
    )

    assert set(summary["group"]) == {"A", "B", "B2", "C"}
    assert bool(gate.loc[gate["k"] == "overall", "gate_pass"].iloc[0])


def test_ewma_scores_use_single_fixed_alpha():
    scores = np.array([10.0, 0.0, 0.0])

    smoothed = ewma_scores(scores, alpha=0.3)

    np.testing.assert_allclose(smoothed, np.array([10.0, 7.0, 4.9]))


def test_cusum_scores_accumulate_positive_robust_residual_excess():
    scores = np.array([1.0, 1.0, 3.0, 4.0])
    reference = np.array([1.0, 1.0, 1.0, 2.0])

    values = cusum_scores(scores, reference=reference, drift=0.5)

    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] > 0.0
    assert values[3] > values[2]


def test_tail_and_peak_tail_scores_use_complete_future_windows():
    scores = np.array([2.0, 4.0, 6.0, 8.0])

    tails = tail_scores(scores, k=2)
    peak_tails = peak_tail_scores(scores, k=2)

    np.testing.assert_allclose(
        tails,
        np.array([5.0, 7.0, np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        peak_tails,
        np.array([10.0, 28.0, np.nan, np.nan]),
        equal_nan=True,
    )


def test_peak_gated_tail_scores_zero_non_candidate_points_and_count_candidates():
    scores = np.array([1.0, 10.0, 2.0, 8.0, 2.0])

    gated = peak_gated_tail_scores(scores, k=2, peak_quantile=0.7)

    assert gated[0] == 0.0
    assert gated[1] > 0.0
    assert gated[2] == 0.0
    assert np.isnan(gated[-1])
    assert candidate_count(gated) == 1


def test_top_n_alarms_select_highest_positive_finite_scores_with_stable_tie_break():
    scores = np.array([0.0, 5.0, 5.0, np.nan, 1.0])

    pred = top_n_alarms(scores, top_n=2)

    np.testing.assert_array_equal(np.flatnonzero(pred), np.array([1, 2]))


def test_top_n_alarms_do_not_fill_budget_with_zero_score_non_candidates():
    scores = np.array([0.0, 5.0, 0.0, np.nan, 0.0])

    pred = top_n_alarms(scores, top_n=3)

    np.testing.assert_array_equal(np.flatnonzero(pred), np.array([1]))


def test_delayed_top_n_alarms_emit_at_confirmation_time():
    scores = np.array([0.0, 10.0, 9.0, np.nan, np.nan])

    pred = delayed_top_n_alarms(scores, k=2, top_n=2, length=6)

    np.testing.assert_array_equal(np.flatnonzero(pred), np.array([3, 4]))
