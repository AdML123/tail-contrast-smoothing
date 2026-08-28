from __future__ import annotations

import numpy as np
import pandas as pd

from hbpc.uncertainty_experiments import run_uncertainty_experiments


def test_run_uncertainty_experiments_writes_rank_and_tau_outputs(tmp_path):
    root = tmp_path / "raw"
    for dataset, normal_tail, anomaly_tail in [("A", 0.2, 1.0), ("B", 0.8, 0.4), ("C", 0.3, 0.7)]:
        scores = np.full(200, 0.1, dtype=float)
        labels = np.zeros(200, dtype=int)
        for idx, start in enumerate([20, 60, 100, 140]):
            if idx % 2 == 0:
                scores[start] = 5.0
                scores[start + 1 : start + 4] = normal_tail
            else:
                scores[start] = 5.0
                scores[start + 1 : start + 12] = anomaly_tail
                labels[start : start + 12] = 1
        run_dir = root / dataset / "one_step" / "0"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "scores.npz", scores=scores, labels=labels, pred=np.zeros_like(labels))

    # Minimal adaptation summary for tau CI and leave-one-dataset-out correlation.
    adapt_dir = tmp_path / "adapt" / "metrics"
    adapt_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["A", "B", "C"],
            "predictor": ["one_step"] * 3,
            "seed": [0, 0, 0],
            "tau_anomaly_median": [5.0, 2.0, 4.0],
            "tau_normal_median": [1.0, 4.0, 1.5],
            "tau_ratio": [5.0, 0.5, 2.7],
            "r_K3": [0.8, -0.3, 0.4],
        }
    ).to_csv(adapt_dir / "adaptation_correlation_rows.csv", index=False)

    rank, tau, corr = run_uncertainty_experiments(
        score_root=root,
        output_dir=tmp_path / "out",
        adaptation_rows_path=adapt_dir / "adaptation_correlation_rows.csv",
        datasets=("A", "B", "C"),
        methods=("one_step",),
        seeds=(0,),
        horizons=(3,),
        n_boot=100,
    )

    assert {"rank_biserial", "ci_low", "ci_high", "p_perm", "p_holm", "p_bh"}.issubset(rank.columns)
    assert {"metric", "estimate", "ci_low", "ci_high"}.issubset(tau.columns)
    assert {"left_out", "spearman_r"}.issubset(corr.columns)
    assert (tmp_path / "out" / "tables" / "rank_biserial_uncertainty.csv").is_file()
    assert (tmp_path / "out" / "tables" / "tau_uncertainty.csv").is_file()
    assert (tmp_path / "out" / "tables" / "correlation_leave_one_out.csv").is_file()
    assert (tmp_path / "out" / "figures" / "rank_biserial_ci_forest.png").is_file()
