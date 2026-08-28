from __future__ import annotations

from pathlib import Path

import numpy as np

from hbpc.supplement_experiments import run_supplement_experiments


def _write_score(path: Path, scores: list[float], labels: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "scores.npz", scores=np.asarray(scores, dtype=float), labels=np.asarray(labels, dtype=int))


def test_run_supplement_experiments_writes_tables_and_figures(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_score(root / "SMD" / "one_step" / "0", [1, 9, 7, 6, 1, 1, 8, 7, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    _write_score(root / "MSL" / "one_step" / "0", [1, 5, 1, 1, 1, 6, 5, 4, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

    all_rows, phenomenon, fairness = run_supplement_experiments(
        score_roots=(root,),
        output_dir=tmp_path / "out",
        datasets=("SMD", "MSL"),
        methods=("one_step",),
        seeds=(0,),
        top_ns=(2, 3),
        windows=(1, 2, 3),
        phenomenon_horizons=(1, 2, 3),
        high_fraction=0.5,
    )

    assert not all_rows.empty
    assert not phenomenon.empty
    assert not fairness.empty
    assert (tmp_path / "out" / "metrics" / "supplement_all_rows.csv").exists()
    assert (tmp_path / "out" / "tables" / "cross_dataset_phenomenon.csv").exists()
    assert (tmp_path / "out" / "tables" / "delay_fairness.csv").exists()
    assert (tmp_path / "out" / "figures" / "budget_curve_smd_one_step.png").exists()
    assert {"raw_delayed", "ewma_delayed", "confirmation_mean"}.issubset(set(all_rows["postprocess"]))
    assert not {"backward_avg", "backward_avg_delayed", "forward_avg"}.intersection(set(all_rows["postprocess"]))
    assert {"raw_delayed", "ewma_delayed"}.issubset(set(fairness["postprocess"]))
    assert "backward_avg_delayed" not in set(fairness["postprocess"])



def test_fast_rank_effect_matches_directional_expectation():
    from hbpc.supplement_experiments import _fast_rank_effect

    effect = _fast_rank_effect(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.2, 1.5]))

    assert effect["n_A"] == 3
    assert effect["n_B"] == 3
    assert effect["rank_biserial"] > 0.9
    assert effect["median_ratio_B_over_A"] > 4.0



def test_run_supplement_experiments_can_limit_benchmark_datasets(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    _write_score(root / "SMD" / "one_step" / "0", [1, 9, 7, 6, 1, 1, 8, 7, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    _write_score(root / "SMAP" / "one_step" / "0", [1, 5, 1, 1, 1, 6, 5, 4, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])

    all_rows, phenomenon, _ = run_supplement_experiments(
        score_roots=(root,),
        output_dir=tmp_path / "out2",
        datasets=("SMD", "SMAP"),
        benchmark_datasets=("SMD",),
        methods=("one_step",),
        seeds=(0,),
        top_ns=(2,),
        windows=(1, 2),
        phenomenon_horizons=(1, 2),
        high_fraction=0.5,
    )

    assert set(all_rows["dataset"]) == {"SMD"}
    assert {"SMD", "SMAP"}.issubset(set(phenomenon["dataset"]))
