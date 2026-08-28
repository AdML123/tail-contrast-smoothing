from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_spl_tables import generate_spl_tables


def test_spl_tables_use_all_six_datasets(tmp_path: Path):
    input_root, output_dir = tmp_path / "inputs", tmp_path / "tables"
    source = input_root / "tables"
    source.mkdir(parents=True)
    datasets = ["SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"]
    pd.DataFrame({"dataset": datasets, "matched_pairs": [20, 20, 13, 20, 7, 31], "peak_smd": [0.01] * 6, "tail_contrast": [0.1] * 6, "contrast_ci_low": [0.0] * 6, "contrast_ci_high": [0.2] * 6, "regime": ["positive", "positive", "underpowered", "positive", "underpowered", "null_or_uncertain"], "evidence_status": ["primary", "primary", "exploratory_underpowered", "primary", "exploratory_underpowered", "primary"]}).to_csv(source / "dataset_tail_contrast.csv", index=False)
    pd.DataFrame({"dataset": [d for d in datasets for _ in range(4)], "method": ["raw_realtime", "raw_delayed", "confirmation_mean", "ewma"] * 6, "raw_f1": [0.1] * 24, "event_recall": [0.2] * 24, "mttd": [1.0] * 24}).to_csv(source / "delay_aware_performance.csv", index=False)
    outputs = generate_spl_tables(input_root=input_root, output_dir=output_dir)
    table1 = (output_dir / "table1_regimes.tex").read_text()
    table2 = (output_dir / "table2_performance.tex").read_text()
    assert all(dataset in table1 and dataset in table2 for dataset in datasets)
    assert all(token in table1 for token in ("Case", "Dataset", "delta", "95\\% CI", "Delta F1"))
    assert "SMAP" in table1 and "SWaT" in table1
    assert "positive" in table1 and "null-compatible" in table1 and "reversal" in table1
    assert all(token in table2 for token in ("F1", "raw", "del.", "conf.", "EWMA", "event", "recall", "MTTD"))
    assert all(token not in table2 for token in ("resizebox", "scalebox", "textwidth", "table*", "|"))
