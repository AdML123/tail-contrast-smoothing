from __future__ import annotations

import json

import pandas as pd

from hbpc.spl_result_verifier import verify_spl_results


def write_minimal_outputs(root, methods=None, selected_by="frozen_protocol"):
    methods = ["raw_realtime", "raw_delayed", "confirmation_mean", "ewma"] if methods is None else methods
    tables = root / "tables"
    audits = root / "audits"
    tables.mkdir(parents=True)
    audits.mkdir(parents=True)
    pd.DataFrame({"dataset": ["SMD"] * len(methods), "method": methods, "raw_f1": [0.1] * len(methods), "event_recall": [0.2] * len(methods), "mttd": [1.0] * len(methods), "selected_by": [selected_by] * len(methods)}).to_csv(tables / "delay_aware_performance.csv", index=False)
    (audits / "artifact_deduplication.json").write_text(json.dumps({"input_artifacts": 1, "unique_artifacts": 1}), encoding="utf-8")
    write_valid_contrast(root)


def write_valid_contrast(root):
    pd.DataFrame({"dataset": ["SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"], "matched_pairs": [20] * 6, "contrast_ci_low": [0.1] * 6, "contrast_ci_high": [0.2] * 6, "regime": ["positive"] * 6, "evidence_status": ["primary"] * 6}).to_csv(root / "tables" / "dataset_tail_contrast.csv", index=False)


def test_verifier_rejects_duplicate_method_rows(tmp_path):
    write_minimal_outputs(tmp_path, methods=["forward_avg", "backward_avg"])
    report = verify_spl_results(tmp_path)
    assert not report["passed"]
    assert "duplicate average comparator" in report["errors"]


def test_verifier_rejects_test_selected_best_rows(tmp_path):
    write_minimal_outputs(tmp_path, selected_by="test_raw_f1")
    report = verify_spl_results(tmp_path)
    assert not report["passed"]
    assert "test-selected protocol" in report["errors"]


def test_verifier_accepts_explicitly_underpowered_exploratory_rows(tmp_path):
    write_minimal_outputs(tmp_path)
    frame = pd.read_csv(tmp_path / "tables" / "dataset_tail_contrast.csv").assign(matched_pairs=[20, 20, 6, 20, 15, 22], regime=["positive", "positive", "underpowered", "positive", "underpowered", "positive"], evidence_status=["primary", "primary", "exploratory_underpowered", "primary", "exploratory_underpowered", "primary"])
    frame.to_csv(tmp_path / "tables" / "dataset_tail_contrast.csv", index=False)
    report = verify_spl_results(tmp_path)
    assert report["passed"]


def test_verifier_rejects_underpowered_primary_regime_label(tmp_path):
    write_minimal_outputs(tmp_path)
    frame = pd.read_csv(tmp_path / "tables" / "dataset_tail_contrast.csv").assign(matched_pairs=[20, 20, 6, 20, 15, 22], regime=["positive", "positive", "null_or_uncertain", "positive", "underpowered", "positive"], evidence_status=["primary", "primary", "exploratory_underpowered", "primary", "exploratory_underpowered", "primary"])
    frame.to_csv(tmp_path / "tables" / "dataset_tail_contrast.csv", index=False)
    report = verify_spl_results(tmp_path)
    assert "underpowered dataset has primary regime label" in report["errors"]
