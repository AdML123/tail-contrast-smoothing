import numpy as np
import pandas as pd

from hbpc.rrp_detector_experiments import run_rrp_detector_pilot


def test_run_rrp_detector_pilot_writes_metrics_best_and_gate(tmp_path):
    scores_root = tmp_path / "scores" / "raw"
    run_dir = scores_root / "SMD" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    scores = np.linspace(0.0, 1.0, 80)
    scores[[10, 30, 50, 55, 60, 70]] = [5.0, 6.0, 10.0, 9.0, 8.0, 7.0]
    labels = np.zeros(80, dtype=int)
    labels[48:62] = 1
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels)

    output_dir = tmp_path / "out"
    rows, best, gate = run_rrp_detector_pilot(
        scores_root=scores_root,
        output_dir=output_dir,
        dataset="SMD",
        detector="one_step",
        seeds=(0,),
        horizons=(3,),
        top_ns=(3, 5),
        peak_quantiles=(0.9,),
    )

    assert {"raw", "ewma", "cusum", "tail_K", "peak_tail_K", "peak_gated_tail_K"} <= set(rows["method"])
    assert {
        "raw_f1",
        "pa_f1",
        "event_recall",
        "mttd",
        "candidate_count",
        "predicted_points_actual",
        "predicted_events",
    } <= set(rows.columns)
    assert {"baseline_raw", "peak_gated_tail_K"} <= set(best["family"])
    assert "gate_pass" in gate.columns
    assert (output_dir / "metrics" / "rrp_detector_rows.csv").exists()
    assert (output_dir / "tables" / "rrp_detector_best.csv").exists()
    assert (output_dir / "tables" / "rrp_detector_gate.csv").exists()
    persisted = pd.read_csv(output_dir / "metrics" / "rrp_detector_rows.csv")
    assert len(persisted) == len(rows)


def test_peak_gated_rows_report_candidate_shortage_without_filling_budget(tmp_path):
    scores_root = tmp_path / "scores" / "raw"
    run_dir = scores_root / "SMD" / "one_step" / "0"
    run_dir.mkdir(parents=True)
    scores = np.ones(30, dtype=float)
    scores[[5, 12]] = [20.0, 10.0]
    labels = np.zeros(30, dtype=int)
    labels[10:15] = 1
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels)

    rows, _best, _gate = run_rrp_detector_pilot(
        scores_root=scores_root,
        output_dir=tmp_path / "out",
        dataset="SMD",
        detector="one_step",
        seeds=(0,),
        horizons=(3,),
        top_ns=(5,),
        peak_quantiles=(0.95,),
    )

    gated = rows[rows["method"] == "peak_gated_tail_K"].iloc[0]
    assert gated["candidate_count"] < gated["top_n"]
    assert gated["predicted_points_actual"] == gated["candidate_count"]
