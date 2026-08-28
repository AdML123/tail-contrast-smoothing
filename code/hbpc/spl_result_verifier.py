from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED_METHODS = {"raw_realtime", "raw_delayed", "confirmation_mean", "ewma"}
FORBIDDEN_METHODS = {"forward_avg", "backward_avg"}
REQUIRED_DATASETS = {"SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"}
MINIMUM_PRIMARY_PAIRS = 20


def verify_spl_results(root: Path | str) -> dict[str, object]:
    root = Path(root)
    errors: list[str] = []
    performance = root / "tables" / "delay_aware_performance.csv"
    contrast_path = root / "tables" / "dataset_tail_contrast.csv"
    audit_path = root / "audits" / "artifact_deduplication.json"
    if not performance.exists():
        errors.append("missing delay-aware performance table")
        return {"passed": False, "errors": errors}
    frame = pd.read_csv(performance)
    methods = set(frame.get("method", pd.Series(dtype=str)).dropna().astype(str))
    forbidden = methods & FORBIDDEN_METHODS
    if forbidden:
        errors.append("duplicate average comparator")
    if not REQUIRED_METHODS <= methods:
        errors.append("missing required method")
    if "selected_by" in frame and any(frame["selected_by"].astype(str) != "frozen_protocol"):
        errors.append("test-selected protocol")
    datasets = set(frame.get("dataset", pd.Series(dtype=str)).dropna().astype(str))
    if datasets and not datasets <= REQUIRED_DATASETS:
        errors.append("unknown dataset")
    for column in ("raw_f1", "event_recall", "mttd"):
        if column in frame and not pd.to_numeric(frame[column], errors="coerce").notna().all():
            errors.append(f"non-finite {column}")
    if not audit_path.exists():
        errors.append("missing duplicate audit")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if int(audit.get("unique_artifacts", 0)) > int(audit.get("input_artifacts", 0)):
            errors.append("invalid duplicate audit")
    if not contrast_path.exists():
        errors.append("missing peak-matched contrast table")
    else:
        contrast = pd.read_csv(contrast_path)
        contrast_datasets = set(contrast.get("dataset", pd.Series(dtype=str)).dropna().astype(str))
        if contrast_datasets != REQUIRED_DATASETS:
            errors.append("contrast table does not cover all six datasets")
        if "matched_pairs" in contrast:
            matched = pd.to_numeric(contrast["matched_pairs"], errors="coerce")
            if matched.isna().any() or (matched <= 0).any():
                errors.append("invalid matched event pair count")
            primary_count = int((matched >= MINIMUM_PRIMARY_PAIRS).sum())
            if primary_count == 0:
                errors.append("no primary dataset meets 20 matched event pairs")
            if "evidence_status" not in contrast:
                errors.append("missing evidence tier")
            else:
                status = contrast["evidence_status"].astype(str)
                expected = matched.map(lambda value: "primary" if value >= MINIMUM_PRIMARY_PAIRS else "exploratory_underpowered")
                if not status.equals(expected):
                    errors.append("inconsistent evidence tier")
                underpowered = matched < MINIMUM_PRIMARY_PAIRS
                if (contrast.loc[underpowered, "regime"].astype(str) != "underpowered").any():
                    errors.append("underpowered dataset has primary regime label")
        for column in ("contrast_ci_low", "contrast_ci_high"):
            if column in contrast and not pd.to_numeric(contrast[column], errors="coerce").notna().all():
                errors.append(f"non-finite {column}")
    return {"passed": not errors, "errors": errors, "checked_methods": sorted(methods), "checked_datasets": sorted(datasets)}
