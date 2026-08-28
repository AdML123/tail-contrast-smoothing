import numpy as np

from hbpc.scar_experiments import run_scar_pilot


def test_run_scar_pilot_writes_expected_tables(tmp_path):
    run_dir = tmp_path / "raw" / "SMD" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    scores = np.array([0, 0, 5, 0, 0, 4, 4, 4, 0, 0, 6, 5, 4, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels, pred=np.zeros_like(labels))

    rows, best, gate = run_scar_pilot(
        scores_root=tmp_path / "raw",
        output_dir=tmp_path / "out",
        seeds=[0],
        top_ns=[3],
        ks=[3],
        n_bins_values=[2],
        q_uppers=[0.9],
        n_refs=[2],
        calibration_fraction=0.5,
        min_bin_count=1,
    )

    assert not rows.empty
    assert not best.empty
    assert not gate.empty
    assert (tmp_path / "out" / "metrics" / "scar_all_rows.csv").exists()
    assert (tmp_path / "out" / "tables" / "scar_best_by_family.csv").exists()
    assert (tmp_path / "out" / "tables" / "scar_gate_report.csv").exists()
    assert (tmp_path / "out" / "tables" / "scar_bin_diagnostics.csv").exists()
    assert (tmp_path / "out" / "tables" / "scar_oracle_report.csv").exists()
