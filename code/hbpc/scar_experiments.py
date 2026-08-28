from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.mars import ewma_score, jaccard_indices, shift_alarm_indices, tail_score, top_n_indices
from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.rrp_experiments import ScoreRun, load_score_run
from hbpc.scar import build_scar_calibration, scar_score


KS = (3, 5)
TOP_NS = (100, 300, 500)
N_BINS_VALUES = (5, 8)
Q_UPPERS = (0.90, 0.95)
N_REFS = (100, 200, 300)


def run_scar_pilot(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-scar-smd-pilot"),
    dataset: str = "SMD",
    detector: str = "one_step",
    seeds: Sequence[int] = (0,),
    ks: Sequence[int] = KS,
    top_ns: Sequence[int] = TOP_NS,
    n_bins_values: Sequence[int] = N_BINS_VALUES,
    q_uppers: Sequence[float] = Q_UPPERS,
    n_refs: Sequence[int] = N_REFS,
    calibration_fraction: float = 0.10,
    min_bin_count: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for seed in seeds:
        run = load_score_run(scores_root / dataset / detector / str(seed))
        for top_n in top_ns:
            rows.extend(_ewma_rows(dataset, detector, int(seed), run, int(top_n)))
        for k in ks:
            tail = tail_score(run.scores, k=int(k))
            tail_bases: dict[int, np.ndarray] = {}
            for top_n in top_ns:
                tail_base = top_n_indices(tail, int(top_n))
                tail_bases[int(top_n)] = tail_base
                rows.append(
                    _metric_row(
                        dataset=dataset,
                        detector=detector,
                        seed=int(seed),
                        setting="baseline",
                        method="tail_K",
                        score=tail,
                        alarm_indices=shift_alarm_indices(tail_base, shift=int(k), length=run.scores.size),
                        labels=run.labels,
                        top_n=int(top_n),
                        k=int(k),
                        n_bins=np.nan,
                        q_upper=np.nan,
                        n_ref=np.nan,
                        calibration_size=0,
                        tail_jaccard=1.0,
                    )
                )

            for n_bins in n_bins_values:
                for q_upper in q_uppers:
                    for n_ref in n_refs:
                        for setting, cal_scores, cal_labels in _calibration_sets(run, calibration_fraction):
                            calibration, diag_rows = build_scar_calibration(
                                cal_scores,
                                k=int(k),
                                n_bins=int(n_bins),
                                q_upper=float(q_upper),
                                n_ref=int(n_ref),
                                min_bin_count=int(min_bin_count),
                                labels=cal_labels,
                            )
                            score = scar_score(run.scores, calibration, k=int(k))
                            diagnostics.extend(
                                {
                                    "dataset": dataset,
                                    "detector": detector,
                                    "seed": int(seed),
                                    "setting": setting,
                                    "k": int(k),
                                    "n_bins": int(n_bins),
                                    "q_upper": float(q_upper),
                                    "n_ref": int(n_ref),
                                    "calibration_size": int(np.isfinite(cal_scores).sum()),
                                    **row,
                                }
                                for row in diag_rows
                            )
                            for top_n in top_ns:
                                base = top_n_indices(score, int(top_n))
                                shifted = shift_alarm_indices(base, shift=int(k), length=run.scores.size)
                                rows.append(
                                    _metric_row(
                                        dataset=dataset,
                                        detector=detector,
                                        seed=int(seed),
                                        setting=setting,
                                        method="scar",
                                        score=score,
                                        alarm_indices=shifted,
                                        labels=run.labels,
                                        top_n=int(top_n),
                                        k=int(k),
                                        n_bins=int(n_bins),
                                        q_upper=float(q_upper),
                                        n_ref=int(n_ref),
                                        calibration_size=int(np.isfinite(cal_scores).sum()),
                                        tail_jaccard=jaccard_indices(base, tail_bases[int(top_n)]),
                                    )
                                )

    frame = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    best = _best_rows(frame)
    gate = _gate_summary(best)
    oracle = _oracle_report(best)
    _write_outputs(output_dir, frame, best, gate, diag, oracle)
    return frame, best, gate


def _calibration_sets(run: ScoreRun, calibration_fraction: float) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    n = run.scores.size
    cal_len = max(1, min(n, int(np.ceil(n * float(calibration_fraction)))))
    labels = np.asarray(run.labels).astype(bool).reshape(-1)
    return (
        ("deployment", run.scores[:cal_len], labels[:cal_len]),
        ("oracle", run.scores[~labels], np.zeros(int((~labels).sum()), dtype=bool)),
    )


def _ewma_rows(dataset: str, detector: str, seed: int, run: ScoreRun, top_n: int) -> list[dict[str, object]]:
    score = ewma_score(run.scores, alpha=0.3)
    return [
        _metric_row(
            dataset=dataset,
            detector=detector,
            seed=seed,
            setting="baseline",
            method="ewma",
            score=score,
            alarm_indices=top_n_indices(score, top_n),
            labels=run.labels,
            top_n=top_n,
            k=0,
            n_bins=np.nan,
            q_upper=np.nan,
            n_ref=np.nan,
            calibration_size=0,
            tail_jaccard=np.nan,
        )
    ]


def _metric_row(
    dataset: str,
    detector: str,
    seed: int,
    setting: str,
    method: str,
    score: np.ndarray,
    alarm_indices: np.ndarray,
    labels: np.ndarray,
    top_n: int,
    k: int,
    n_bins: float,
    q_upper: float,
    n_ref: float,
    calibration_size: int,
    tail_jaccard: float,
) -> dict[str, object]:
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    pred = np.zeros(labels_arr.size, dtype=np.int64)
    valid_indices = np.asarray(alarm_indices, dtype=np.int64)
    valid_indices = valid_indices[(valid_indices >= 0) & (valid_indices < pred.size)]
    pred[valid_indices] = 1
    adjusted = point_adjust(pred, labels_arr)
    pred_events = events_from_binary(pred)
    return {
        "dataset": dataset,
        "detector": detector,
        "seed": int(seed),
        "setting": setting,
        "method": method,
        "top_n": int(top_n),
        "k": int(k),
        "n_bins": int(n_bins) if np.isfinite(n_bins) else np.nan,
        "q_upper": float(q_upper) if np.isfinite(q_upper) else np.nan,
        "n_ref": int(n_ref) if np.isfinite(n_ref) else np.nan,
        "calibration_size": int(calibration_size),
        "raw_f1": f1(pred, labels_arr),
        "pa_f1": f1(adjusted, labels_arr),
        "event_recall": _event_recall(pred, labels_arr),
        "mttd": detection_delay(pred, labels_arr),
        "candidate_count": _candidate_count(score),
        "predicted_points": int(top_n),
        "predicted_points_actual": int(pred.sum()),
        "predicted_events": len(pred_events),
        "tail_jaccard": float(tail_jaccard) if np.isfinite(tail_jaccard) else np.nan,
    }


def _event_recall(pred: np.ndarray, labels: np.ndarray) -> float:
    pred_arr = np.asarray(pred).astype(bool).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    events = events_from_binary(labels_arr)
    if not events:
        return 0.0
    hits = sum(bool(pred_arr[start : stop + 1].any()) for start, stop in events)
    return float(hits / len(events))


def _candidate_count(score: np.ndarray) -> int:
    arr = np.asarray(score, dtype=np.float64).reshape(-1)
    return int(np.isfinite(arr).sum())


def _best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["family"] = frame["setting"].astype(str) + "_" + frame["method"].astype(str)
    families = ["baseline_ewma", "baseline_tail_K", "deployment_scar", "oracle_scar"]
    rows: list[pd.Series] = []
    for family in families:
        subset = frame[frame["family"] == family]
        if subset.empty:
            continue
        idx = subset.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).index[0]
        rows.append(subset.loc[idx].copy())
    return pd.DataFrame(rows)


def _gate_summary(best: pd.DataFrame) -> pd.DataFrame:
    baseline = best[best["family"].isin(["baseline_ewma", "baseline_tail_K"])]
    scar = best[best["family"] == "deployment_scar"]
    if baseline.empty or scar.empty:
        return pd.DataFrame(columns=_gate_columns())
    baseline_row = baseline.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).iloc[0]
    scar_row = scar.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).iloc[0]
    raw_f1_pass = bool(scar_row["raw_f1"] >= baseline_row["raw_f1"] + 0.02)
    pa_f1_pass = bool(scar_row["pa_f1"] >= baseline_row["pa_f1"] + 0.01)
    recall_pass = bool(scar_row["event_recall"] >= baseline_row["event_recall"] - 0.03)
    mttd_pass = _mttd_pass(scar_row["mttd"], baseline_row["mttd"], int(scar_row["k"]))
    jaccard_pass = bool(scar_row["tail_jaccard"] <= 0.90)
    gate_pass = bool(raw_f1_pass and pa_f1_pass and recall_pass and mttd_pass and jaccard_pass)
    return pd.DataFrame(
        [
            {
                "baseline_family": baseline_row["family"],
                "baseline_method": baseline_row["method"],
                "baseline_raw_f1": baseline_row["raw_f1"],
                "baseline_pa_f1": baseline_row["pa_f1"],
                "baseline_event_recall": baseline_row["event_recall"],
                "baseline_mttd": baseline_row["mttd"],
                "scar_raw_f1": scar_row["raw_f1"],
                "raw_f1_pass": raw_f1_pass,
                "scar_pa_f1": scar_row["pa_f1"],
                "pa_f1_pass": pa_f1_pass,
                "scar_event_recall": scar_row["event_recall"],
                "event_recall_pass": recall_pass,
                "scar_mttd": scar_row["mttd"],
                "scar_k": int(scar_row["k"]),
                "mttd_pass": mttd_pass,
                "scar_tail_jaccard": scar_row["tail_jaccard"],
                "jaccard_pass": jaccard_pass,
                "scar_top_n": int(scar_row["top_n"]),
                "scar_n_bins": int(scar_row["n_bins"]),
                "scar_q_upper": scar_row["q_upper"],
                "scar_n_ref": int(scar_row["n_ref"]),
                "gate_pass": gate_pass,
            }
        ]
    )


def _oracle_report(best: pd.DataFrame) -> pd.DataFrame:
    baseline = best[best["family"].isin(["baseline_ewma", "baseline_tail_K"])]
    oracle = best[best["family"] == "oracle_scar"]
    if baseline.empty or oracle.empty:
        return pd.DataFrame()
    baseline_row = baseline.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).iloc[0]
    oracle_row = oracle.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).iloc[0]
    return pd.DataFrame(
        [
            {
                "baseline_family": baseline_row["family"],
                "baseline_raw_f1": baseline_row["raw_f1"],
                "baseline_pa_f1": baseline_row["pa_f1"],
                "baseline_event_recall": baseline_row["event_recall"],
                "oracle_raw_f1": oracle_row["raw_f1"],
                "oracle_pa_f1": oracle_row["pa_f1"],
                "oracle_event_recall": oracle_row["event_recall"],
                "oracle_mttd": oracle_row["mttd"],
                "oracle_tail_jaccard": oracle_row["tail_jaccard"],
                "oracle_k": int(oracle_row["k"]),
                "oracle_top_n": int(oracle_row["top_n"]),
                "oracle_n_bins": int(oracle_row["n_bins"]),
                "oracle_q_upper": oracle_row["q_upper"],
                "oracle_n_ref": int(oracle_row["n_ref"]),
                "oracle_raw_f1_pass": bool(oracle_row["raw_f1"] >= baseline_row["raw_f1"] + 0.02),
                "oracle_pa_f1_pass": bool(oracle_row["pa_f1"] >= baseline_row["pa_f1"] + 0.01),
            }
        ]
    )


def _mttd_pass(scar_mttd: float, baseline_mttd: float, k: int) -> bool:
    scar_nan = pd.isna(scar_mttd)
    baseline_nan = pd.isna(baseline_mttd)
    if scar_nan and baseline_nan:
        return True
    if scar_nan:
        return False
    if baseline_nan:
        return True
    return bool(scar_mttd <= baseline_mttd + k)


def _gate_columns() -> list[str]:
    return [
        "baseline_family",
        "baseline_method",
        "baseline_raw_f1",
        "baseline_pa_f1",
        "baseline_event_recall",
        "baseline_mttd",
        "scar_raw_f1",
        "raw_f1_pass",
        "scar_pa_f1",
        "pa_f1_pass",
        "scar_event_recall",
        "event_recall_pass",
        "scar_mttd",
        "scar_k",
        "mttd_pass",
        "scar_tail_jaccard",
        "jaccard_pass",
        "scar_top_n",
        "scar_n_bins",
        "scar_q_upper",
        "scar_n_ref",
        "gate_pass",
    ]


def _write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    best: pd.DataFrame,
    gate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(metrics_dir / "scar_all_rows.csv", index=False)
    best.to_csv(tables_dir / "scar_best_by_family.csv", index=False)
    gate.to_csv(tables_dir / "scar_gate_report.csv", index=False)
    diagnostics.to_csv(tables_dir / "scar_bin_diagnostics.csv", index=False)
    oracle.to_csv(tables_dir / "scar_oracle_report.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SCAR SMD pilot.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-scar-smd-pilot"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detector", default="one_step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(TOP_NS))
    parser.add_argument("--n-bins-values", nargs="+", type=int, default=list(N_BINS_VALUES))
    parser.add_argument("--q-uppers", nargs="+", type=float, default=list(Q_UPPERS))
    parser.add_argument("--n-refs", nargs="+", type=int, default=list(N_REFS))
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-bin-count", type=int, default=30)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_scar_pilot(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detector=args.detector,
        seeds=tuple(args.seeds),
        ks=tuple(args.ks),
        top_ns=tuple(args.top_ns),
        n_bins_values=tuple(args.n_bins_values),
        q_uppers=tuple(args.q_uppers),
        n_refs=tuple(args.n_refs),
        calibration_fraction=float(args.calibration_fraction),
        min_bin_count=int(args.min_bin_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
