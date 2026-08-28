from pathlib import Path

from hbpc.alignment_experiments import run_alignment_benchmark


def test_run_alignment_benchmark_writes_inventory_and_tables(tmp_path):
    root = Path(r"D:/HBPC/results-smd-gate/raw")
    if not (root / "SMD" / "one_step" / "0" / "scores.npz").exists():
        return
    all_rows, best, inventory = run_alignment_benchmark(
        scores_root=root,
        output_dir=tmp_path,
        dataset="SMD",
        methods=("one_step",),
        seeds=(0,),
    )
    assert not all_rows.empty
    assert not best.empty
    assert {"raw", "forward_avg", "ewma"}.issubset(set(all_rows["postprocess"]))
    assert (tmp_path / "tables" / "predictor_inventory.csv").exists()
    assert "OmniAnomaly" in set(inventory["predictor"])
