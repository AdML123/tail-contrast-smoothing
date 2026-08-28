from pathlib import Path
import subprocess
import sys

import pandas as pd

from hbpc.paper_result_verifier import EXPECTED, MAJOR_REVISION_EXPECTED, verify_paper_results


def test_verify_paper_results_checks_core_tables(tmp_path: Path):
    root = tmp_path / "results"
    (root / "results-strr-five-dataset" / "tables").mkdir(parents=True)
    (root / "results-adaptation-causal-score-artifact" / "tables").mkdir(parents=True)

    phenomenon = []
    for row in EXPECTED["phenomenon"]:
        phenomenon.append({**row, "seed": -1})
    pd.DataFrame(phenomenon).to_csv(root / "results-strr-five-dataset" / "tables" / "cross_dataset_phenomenon.csv", index=False)
    pd.DataFrame([
        {"dataset": "SMD", "top_n": 300, "postprocess": "raw", "k": 0, "raw_f1": 0.12206572769953054, "pa_f1": 0.6874487284659558, "event_recall": 0.6428571428571429, "event_precision": 0.07116104868913857, "mttd": 15.5555555555},
    ]).to_csv(root / "results-strr-five-dataset" / "tables" / "budget_curve_summary.csv", index=False)
    pd.DataFrame([
        {"dataset": "SMD", "top_n": 300, "postprocess": "forward_avg", "k": 3, "raw_f1": 0.37089201877934275, "pa_f1": 0.8669387755102042, "event_recall": 0.8214285714285715, "event_precision": 0.421875, "event_f1": 0.5574506283662477, "mttd": 2.347826086956522},
    ]).to_csv(root / "results-strr-five-dataset" / "tables" / "delay_fairness.csv", index=False)
    pd.DataFrame([
        {"dataset": "SMD", "tau_anomaly_median": 0.7384624969221785, "tau_normal_median": 0.008085403982021074, "tau_ratio": 91.33278913017135},
    ]).to_csv(root / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_summary.csv", index=False)
    pd.DataFrame([
        {"feature": "tau_normal_median", "target": "r_K3", "n": 5, "spearman_r": -0.8999999999999998, "spearman_p": 0.03738607346849874},
    ]).to_csv(root / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_correlation_summary.csv", index=False)

    report = verify_paper_results(root, required_groups=("phenomenon",))

    assert report["passed"] is True
    assert report["checked"] == len(EXPECTED["phenomenon"])
    assert report["failures"] == []


def test_verify_paper_results_reports_missing_required_group(tmp_path: Path):
    report = verify_paper_results(tmp_path, required_groups=("public_deep",))

    assert report["passed"] is False
    assert report["failures"]


def test_major_revision_expected_groups_are_verified(tmp_path: Path):
    root = tmp_path / "results"
    (root / "results-synthetic-regime" / "tables").mkdir(parents=True)
    (root / "results-uncertainty" / "tables").mkdir(parents=True)
    (root / "results-sensitivity" / "tables").mkdir(parents=True)
    (root / "results-swat-filter" / "tables").mkdir(parents=True)

    pd.DataFrame(MAJOR_REVISION_EXPECTED["synthetic"]).to_csv(
        root / "results-synthetic-regime" / "tables" / "synthetic_regime_summary.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["synthetic_delta"]).to_csv(
        root / "results-synthetic-regime" / "tables" / "synthetic_delta_summary.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["uncertainty"]).to_csv(
        root / "results-uncertainty" / "tables" / "rank_biserial_uncertainty.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["tau_uncertainty"]).to_csv(
        root / "results-uncertainty" / "tables" / "tau_uncertainty.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["loo"]).to_csv(
        root / "results-uncertainty" / "tables" / "correlation_leave_one_out.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["sensitivity"]).to_csv(
        root / "results-sensitivity" / "tables" / "sensitivity_summary.csv", index=False
    )
    pd.DataFrame(MAJOR_REVISION_EXPECTED["swat_highpass"]).to_csv(
        root / "results-swat-filter" / "tables" / "swat_highpass_summary.csv", index=False
    )

    report = verify_paper_results(
        root,
        required_groups=("synthetic", "uncertainty", "sensitivity", "swat_highpass"),
    )

    assert report["passed"] is True
    assert report["checked"] == sum(len(MAJOR_REVISION_EXPECTED[k]) for k in ["synthetic", "synthetic_delta", "uncertainty", "tau_uncertainty", "loo", "sensitivity", "swat_highpass"])
    assert report["failures"] == []


def test_major_revision_public_deep_best_rows_are_verified(tmp_path: Path):
    root = tmp_path / "results"
    (root / "tables").mkdir(parents=True)
    pd.DataFrame(MAJOR_REVISION_EXPECTED["public_deep"]).to_csv(
        root / "tables" / "public_deep_best_mean_std.csv", index=False
    )

    report = verify_paper_results(root, required_groups=("public_deep",))

    assert report["passed"] is True
    assert report["checked"] == len(MAJOR_REVISION_EXPECTED["public_deep"])
    assert report["failures"] == []


def test_verify_script_is_directly_executable_help():
    result = subprocess.run(
        [sys.executable, "scripts/verify_paper_results.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_verify_script_can_require_major_revision_groups():
    source = Path(__file__).resolve().parents[1] / "hbpc" / "paper_result_verifier.py"
    text = source.read_text(encoding="utf-8")

    assert "--require-major-revision" in text
    assert "synthetic" in text
    assert "swat_highpass" in text