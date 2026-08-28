from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.generate_spl_tables import generate_spl_tables


def test_spl_tables_use_all_six_datasets(tmp_path: Path):
    input_root, output_dir = tmp_path / "inputs", tmp_path / "tables"
    source = input_root / "tables"
    source.mkdir(parents=True)
    datasets = ["SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI"]
    pd.DataFrame({"dataset": datasets, "matched_pairs": [238, 26, 13, 35, 7, 31], "peak_smd": [0.01] * 6, "tail_contrast": [1.0, -1.0, 0.2, 0.8, -0.5, 0.05], "contrast_ci_low": [0.5, -1.5, -0.2, 0.3, -1.0, -0.1], "contrast_ci_high": [1.5, -0.5, 0.6, 1.3, 0.0, 0.2], "regime": ["positive", "reversal", "underpowered", "positive", "underpowered", "null_or_uncertain"], "evidence_status": ["primary", "primary", "exploratory_underpowered", "primary", "exploratory_underpowered", "primary"]}).to_csv(source / "dataset_tail_contrast.csv", index=False)
    f1 = {
        "SMD": [0.07, 0.08, 0.10, 0.09], "MSL": [0.04, 0.03, 0.02, 0.03],
        "SMAP": [0.01, 0.01, 0.01, 0.01], "PSM": [0.20, 0.19, 0.21, 0.20],
        "SWaT": [0.03, 0.02, 0.01, 0.02], "HAI": [0.15, 0.14, 0.14, 0.15],
    }
    pd.DataFrame([
        {"dataset": dataset, "method": method, "raw_f1": f1[dataset][i], "event_recall": 0.2 + i / 100, "mttd": 1.0 + i}
        for dataset in datasets
        for i, method in enumerate(["raw_realtime", "raw_delayed", "confirmation_mean", "ewma"])
    ]).to_csv(source / "delay_aware_performance.csv", index=False)
    outputs = generate_spl_tables(input_root=input_root, output_dir=output_dir)
    table1 = (output_dir / "table1_regimes.tex").read_text()
    table2 = (output_dir / "table2_performance.tex").read_text()
    assert all(dataset in table1 and dataset in table2 for dataset in datasets)
    assert all(token in table1 for token in ("Dataset", "m", "delta", "95\\% CI", "Pred.", "Delta F1", "Tier"))
    assert "SMAP" in table1 and "SWaT" in table1
    assert "positive" in table1 and "null-compatible" in table1 and "reversal" in table1
    assert "E" in table1 and "P" in table1
    assert "Obs." not in table1
    assert all(token in table2 for token in ("F1", "raw", "del.", "conf.", "EWMA", "event", "recall", "MTTD"))
    assert all(token not in table2 for token in ("resizebox", "scalebox", "textwidth", "table*", "|"))
    assert all(token not in table1 for token in ("F1\\raw", "event\\recall", "MTTD"))
    assert all(token not in table2 for token in ("delta_", "95\\% CI", "Delta F1", "Pred.", "Tier"))
    assert "1.000" in table1 and "[0.500, 1.500]" in table1 and "0.020" in table1
