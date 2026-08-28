from __future__ import annotations

import numpy as np

from hbpc.score_benchmark import ScoreRun
from hbpc.swat_filter_experiments import highpass_scores, run_swat_filter_experiment, run_swat_intervention


def test_highpass_scores_removes_slow_baseline_and_keeps_transients():
    t = np.arange(200, dtype=float)
    scores = 2.0 + 0.005 * t
    scores[100] += 4.0

    filtered = highpass_scores(scores, window=51)

    assert filtered[100] > filtered[20]
    assert filtered.min() >= 0.0


def test_run_swat_filter_experiment_writes_regime_movement(tmp_path):
    root = tmp_path / "raw"
    scores = np.full(260, 1.0, dtype=float)
    labels = np.zeros(260, dtype=int)
    labels[160:210] = 1
    scores += np.linspace(0.0, 2.0, 260)
    scores[30:90] += 2.0
    scores[160:210] += np.linspace(3.0, 1.0, 50)
    run_dir = root / "SWaT" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "scores.npz", scores=scores, labels=labels, pred=np.zeros_like(labels))

    table = run_swat_filter_experiment(
        score_root=root,
        output_dir=tmp_path / "out",
        seeds=(0,),
        windows=(31, 61),
        horizons=(3,),
        top_n=20,
    )

    assert {"filter", "filter_window", "r_K3", "tau_normal_median", "forward_gain"}.issubset(table.columns)
    assert set(table["filter"]) == {"none", "highpass"}
    assert (tmp_path / "out" / "tables" / "swat_highpass_summary.csv").is_file()
    assert (tmp_path / "out" / "figures" / "swat_highpass_regime_movement.png").is_file()


def test_swat_intervention_reports_every_window_and_negative_control(tmp_path):
    scores = np.sin(np.arange(80) / 4.0) + 2.0
    labels = np.zeros(80, dtype=bool)
    labels[20:28] = True
    labels[55:64] = True
    run = ScoreRun(dataset="SWaT", predictor="p", seed=0, scores=scores, labels=labels)
    frame = run_swat_intervention(run, training_normal_scores=np.linspace(0.1, 2.0, 50), output_dir=tmp_path, windows=(3, 5, 10), n_boot=100, seed=47)
    assert set(frame["window"]) == {3, 5, 10}
    assert set(frame["analysis"]) == {"observed", "block_permuted_control"}
    assert {"delta_change", "ci_low", "ci_high"} <= set(frame.columns)
