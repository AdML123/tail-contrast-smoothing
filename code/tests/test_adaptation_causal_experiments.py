from __future__ import annotations

import numpy as np
import pandas as pd

from hbpc.adaptation_causal_experiments import (
    run_adaptation_correlation_from_arrays,
    run_adaptation_correlation_from_score_root,
)


def test_run_adaptation_correlation_from_arrays_writes_outputs(tmp_path) -> None:
    errors_a = np.zeros((80, 3), dtype=float)
    labels_a = np.zeros(80, dtype=bool)
    labels_a[30:45] = True
    errors_a[10:15, 0] = [8, 3, 1.5, 1, 0.7]
    errors_a[30:45, :] = np.linspace(7.0, 3.0, 15)[:, None]

    errors_b = np.zeros((80, 3), dtype=float)
    labels_b = np.zeros(80, dtype=bool)
    labels_b[30:45] = True
    errors_b[10:20, 0] = np.linspace(8.0, 4.0, 10)
    errors_b[30:45, :] = np.linspace(8.0, 1.0, 15)[:, None]

    rows, corr = run_adaptation_correlation_from_arrays(
        runs=[
            ("A", "toy", 0, errors_a, labels_a),
            ("B", "toy", 0, errors_b, labels_b),
        ],
        output_dir=tmp_path,
        horizons=(3, 5),
    )

    assert {"dataset", "predictor", "tau_ratio", "r_K3", "r_K5"} <= set(rows.columns)
    assert {"feature", "target", "spearman_r", "pearson_r"} <= set(corr.columns)
    assert (tmp_path / "metrics" / "adaptation_correlation_rows.csv").is_file()
    assert (tmp_path / "tables" / "adaptation_correlation_summary.csv").is_file()
    assert (tmp_path / "metrics" / "adaptation_correlation.json").is_file()
    assert (tmp_path / "figures" / "adaptation_tau_vs_r.png").is_file()

    persisted = pd.read_csv(tmp_path / "metrics" / "adaptation_correlation_rows.csv")
    assert len(persisted) == len(rows)


def test_run_adaptation_correlation_from_score_root_uses_npz_scores(tmp_path) -> None:
    scores = np.full(100, 0.1, dtype=float)
    labels = np.zeros(100, dtype=int)
    labels[60:90] = 1
    scores[10] = 10.0
    scores[60:80] = 8.0

    run_dir = tmp_path / "raw" / "Toy" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "scores.npz", scores=scores, pred=np.zeros_like(labels), labels=labels)

    rows, corr = run_adaptation_correlation_from_score_root(
        score_root=tmp_path / "raw",
        output_dir=tmp_path / "out",
        datasets=("Toy",),
        methods=("one_step",),
        seeds=(0,),
        horizons=(3,),
        high_fraction=0.01,
    )

    assert len(rows) == 1
    assert rows.iloc[0]["source"] == "score_artifact"
    assert rows.iloc[0]["r_K3"] > 0.5
    assert {"feature", "target", "spearman_r", "pearson_r"} <= set(corr.columns)
    assert (tmp_path / "out" / "tables" / "adaptation_dataset_summary.csv").is_file()
    assert (tmp_path / "out" / "tables" / "adaptation_dataset_correlation_summary.csv").is_file()
    assert (tmp_path / "out" / "figures" / "adaptation_tau_vs_r.png").is_file()
