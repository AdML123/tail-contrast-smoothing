import numpy as np
import pandas as pd

from hbpc.rrp_experiments import run_rrp_phenomenon_pilot


def test_run_rrp_phenomenon_pilot_writes_tables_and_figures(tmp_path):
    scores_root = tmp_path / "scores" / "raw"
    run_dir = scores_root / "SMD" / "one_step" / "0"
    run_dir.mkdir(parents=True)

    scores = np.linspace(1.0, 4.0, 80, dtype=float)
    scores[[5, 15, 25, 35, 45, 55, 65, 75]] = [5, 6, 7, 8, 20, 22, 24, 26]
    labels = np.zeros(80, dtype=int)
    labels[40:70] = 1
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels)

    output_dir = tmp_path / "out"
    summary, effects, gate = run_rrp_phenomenon_pilot(
        scores_root=scores_root,
        output_dir=output_dir,
        dataset="SMD",
        detector="one_step",
        seeds=(0,),
        horizons=(3, 5),
        high_fraction=0.25,
        min_group_size=1,
        min_pass_horizons=1,
    )

    assert {"A", "B", "B2", "C"} <= set(summary["group"])
    assert {"A_vs_B", "A_vs_B2"} <= set(effects["comparison"])
    assert "overall" in set(gate["k"].astype(str))
    assert (output_dir / "tables" / "rrp_group_summary.csv").exists()
    assert (output_dir / "tables" / "rrp_effects.csv").exists()
    assert (output_dir / "tables" / "rrp_gate.csv").exists()
    assert (output_dir / "figures" / "relax_ecdf_K3.png").exists()
    assert (output_dir / "figures" / "relaxation_curve_K3.png").exists()
    persisted = pd.read_csv(output_dir / "tables" / "rrp_group_summary.csv")
    assert len(persisted) == len(summary)
