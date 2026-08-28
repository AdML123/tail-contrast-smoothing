from __future__ import annotations

import numpy as np
import pandas as pd

from hbpc.statistical_inference import (
    adjust_pvalues,
    bootstrap_ci,
    bootstrap_rank_biserial_ci,
    leave_one_out_spearman,
    permutation_test_rank_biserial,
    rank_biserial,
)


def test_rank_biserial_is_positive_when_second_group_is_larger():
    first = np.array([0.1, 0.2, 0.3])
    second = np.array([1.0, 1.1, 1.2])

    effect = rank_biserial(first, second)

    assert effect > 0.99


def test_bootstrap_rank_biserial_ci_contains_observed_effect():
    first = np.array([0.1, 0.2, 0.3, 0.4])
    second = np.array([0.8, 0.9, 1.0, 1.1])

    result = bootstrap_rank_biserial_ci(first, second, n_boot=200, seed=7)

    assert result["n_first"] == 4
    assert result["n_second"] == 4
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["estimate"] > 0.9


def test_permutation_test_rank_biserial_reports_small_p_for_separated_groups():
    first = np.array([0.1, 0.2, 0.3])
    second = np.array([1.0, 1.1, 1.2])

    result = permutation_test_rank_biserial(first, second, alternative="greater")

    assert result["estimate"] > 0.99
    assert 0.0 < result["p_value"] <= 0.1
    assert result["n_permutations"] == 20


def test_adjust_pvalues_supports_holm_and_bh():
    p_values = [0.001, 0.02, 0.04]

    holm = adjust_pvalues(p_values, method="holm")
    bh = adjust_pvalues(p_values, method="bh")

    assert holm[0] <= holm[1] <= holm[2]
    assert bh[0] <= bh[1] <= bh[2]
    assert all(0.0 <= value <= 1.0 for value in holm)
    assert all(0.0 <= value <= 1.0 for value in bh)


def test_leave_one_out_spearman_returns_one_row_per_dataset():
    frame = pd.DataFrame(
        {
            "dataset": ["A", "B", "C", "D"],
            "tau_norm": [4.0, 3.0, 2.0, 1.0],
            "r": [0.8, 0.5, 0.1, -0.2],
        }
    )

    out = leave_one_out_spearman(frame, feature="tau_norm", target="r")

    assert list(out["left_out"]) == ["A", "B", "C", "D"]
    assert out["spearman_r"].notna().all()


def test_bootstrap_ci_for_median_is_reproducible():
    values = np.array([1.0, 2.0, 3.0, 4.0, 20.0])

    first = bootstrap_ci(values, statistic=np.median, n_boot=100, seed=3)
    second = bootstrap_ci(values, statistic=np.median, n_boot=100, seed=3)

    assert first == second
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
