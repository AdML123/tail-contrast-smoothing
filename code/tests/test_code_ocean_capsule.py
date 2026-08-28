from __future__ import annotations

from pathlib import Path

from scripts import reproduce_code_ocean_capsule
from scripts.reproduce_code_ocean_capsule import expected_outputs


def test_expected_outputs_cover_paper_artifacts() -> None:
    outputs = set(expected_outputs())
    required = {
        "figures/mechanism_relaxation_toy.png",
        "figures/mechanism_forward_average.png",
        "figures/case_comparison_smd_one_step.png",
        "figures/cross_dataset_rank_biserial.png",
        "figures/tau_norm_vs_r.png",
        "figures/relax_ecdf_K3.png",
        "figures/relax_ecdf_K5.png",
        "figures/relax_ecdf_K10.png",
        "figures/relax_ecdf_K20.png",
        "figures/relaxation_curve_K3.png",
        "figures/relaxation_curve_K5.png",
        "figures/relaxation_curve_K10.png",
        "figures/relaxation_curve_K20.png",
        "figures/budget_curve_smd_one_step.png",
        "figures/window_sensitivity_smd.png",
        "figures/budget_curve_smd_AnomalyTransformer.png",
        "figures/budget_curve_smd_Autoformer.png",
        "figures/budget_curve_smd_TimesNet.png",
        "figures/budget_curve_smd_Transformer.png",
        "figures/synthetic_delta_gain_heatmap.png",
        "figures/rank_biserial_ci_forest.png",
        "figures/sensitivity_k_topn_curves.png",
        "figures/swat_highpass_regime_movement.png",
        "tables/notation.csv",
        "tables/scope.csv",
        "tables/cross_dataset_phenomenon.csv",
        "tables/budget_curve_summary.csv",
        "tables/delay_fairness.csv",
        "tables/adaptation_dataset_summary.csv",
        "tables/adaptation_dataset_correlation_summary.csv",
        "tables/public_deep_budget_curve_summary.csv",
        "tables/public_deep_best_mean_std.csv",
        "tables/negative_variants.csv",
        "tables/synthetic_regime_summary.csv",
        "tables/synthetic_delta_gain_heatmap.csv",
        "tables/synthetic_delta_summary.csv",
        "tables/rank_biserial_uncertainty.csv",
        "tables/tau_uncertainty.csv",
        "tables/correlation_leave_one_out.csv",
        "tables/sensitivity_summary.csv",
        "tables/swat_highpass_summary.csv",
        "capsule_manifest.csv",
        "capsule_verification.json",
    }
    assert required <= outputs


def test_reproducer_wires_optional_full_split_sensitivity() -> None:
    source = Path(reproduce_code_ocean_capsule.__file__).read_text(encoding="utf-8")

    assert "results-five-dataset-one-step-full" in source
    assert "--full-score-root" in source
    assert "full_one_step_scores.exists()" in source