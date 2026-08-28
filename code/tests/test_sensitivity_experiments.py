from __future__ import annotations

import numpy as np

from hbpc.sensitivity_experiments import robust_normalize_scores, run_sensitivity_experiments


def test_robust_normalize_scores_uses_median_and_mad():
    scores = np.array([1.0, 2.0, 3.0, 100.0])

    normalized = robust_normalize_scores(scores)

    assert normalized[3] > normalized[2] >= normalized[1] >= normalized[0]
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0.0


def test_run_sensitivity_experiments_writes_summary_outputs(tmp_path):
    root = tmp_path / "raw"
    scores = np.full(220, 0.1, dtype=float)
    labels = np.zeros(220, dtype=int)
    labels[80:115] = 1
    scores[20] = 8.0
    scores[21:26] = [1.0, 0.5, 0.2, 0.1, 0.1]
    scores[80:100] = 4.0
    run_dir = root / "Toy" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    np.savez(run_dir / "scores.npz", scores=scores, labels=labels, pred=np.zeros_like(labels))

    benchmark, phenomenon, summary = run_sensitivity_experiments(
        score_root=root,
        output_dir=tmp_path / "out",
        datasets=("Toy",),
        methods=("one_step",),
        seeds=(0,),
        windows=(3, 5),
        top_ns=(10, 20),
        high_fractions=(0.01, 0.05),
    )

    assert {"normalization", "top_n", "k", "raw_f1"}.issubset(benchmark.columns)
    assert {"high_fraction", "rank_biserial"}.issubset(phenomenon.columns)
    assert {"best_forward_raw_f1", "best_raw_raw_f1", "best_forward_gain"}.issubset(summary.columns)
    assert (tmp_path / "out" / "tables" / "sensitivity_summary.csv").is_file()
    assert (tmp_path / "out" / "figures" / "sensitivity_k_topn_curves.png").is_file()


def test_run_sensitivity_experiments_tracks_capped_and_full_score_roots(tmp_path):
    def write_run(root, scale):
        scores = np.full(180, 0.1, dtype=float)
        labels = np.zeros(180, dtype=int)
        labels[80:100] = 1
        scores[20] = 8.0
        scores[21:26] = [1.0, 0.5, 0.2, 0.1, 0.1]
        scores[80:100] = float(scale)
        run_dir = root / "Toy" / "one_step" / "0"
        run_dir.mkdir(parents=True)
        np.savez(run_dir / "scores.npz", scores=scores, labels=labels, pred=np.zeros_like(labels))

    capped = tmp_path / "capped"
    full = tmp_path / "full"
    write_run(capped, scale=3.0)
    write_run(full, scale=5.0)

    benchmark, phenomenon, summary = run_sensitivity_experiments(
        score_roots={"capped": capped, "full": full},
        output_dir=tmp_path / "out",
        datasets=("Toy",),
        methods=("one_step",),
        seeds=(0,),
        windows=(3,),
        top_ns=(10,),
        high_fractions=(0.01,),
    )

    assert set(benchmark["split_scope"]) == {"capped", "full"}
    assert set(phenomenon["split_scope"]) == {"capped", "full"}
    assert set(summary["split_scope"]) == {"capped", "full"}