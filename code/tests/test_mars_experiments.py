import numpy as np
import pandas as pd

from hbpc.mars_experiments import run_mars_pilot


def test_run_mars_pilot_writes_rows_best_and_gate(tmp_path):
    scores_root = tmp_path / "scores" / "raw"
    run_dir = scores_root / "SMD" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    scores = np.ones(80, dtype=float)
    scores[10:14] = [1.0, 5.0, 8.0, 10.0]
    scores[45:50] = [3.0, 5.0, 7.0, 8.0, 9.0]
    labels = np.zeros(80, dtype=int)
    labels[44:52] = 1
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels)

    output_dir = tmp_path / "out"
    rows, best, gate = run_mars_pilot(
        scores_root=scores_root,
        output_dir=output_dir,
        seeds=(0,),
        horizons=(3,),
        top_ns=(3, 5),
        alphas=(0.5, 1.0),
    )

    assert {"raw", "ewma", "tail_K", "mars_abs", "mars_rel"} <= set(rows["method"])
    assert {"raw", "ewma", "tail_K", "mars_abs", "mars_rel"} <= set(best["family"])
    assert {
        "baseline_raw_f1",
        "mars_raw_f1",
        "raw_f1_pass",
        "jaccard_pass",
        "gate_pass",
    } <= set(gate.columns)
    assert (output_dir / "metrics" / "mars_rows.csv").exists()
    assert (output_dir / "tables" / "mars_best.csv").exists()
    assert (output_dir / "tables" / "mars_gate.csv").exists()
    persisted = pd.read_csv(output_dir / "metrics" / "mars_rows.csv")
    assert len(persisted) == len(rows)
