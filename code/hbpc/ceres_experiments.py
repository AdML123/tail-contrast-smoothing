from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.ceres import (
    build_tail_envelopes,
    ceres_envelope_score,
    ceres_lite_score,
    normalize_scores,
    robust_location_scale,
)
from hbpc.mars import ewma_score, jaccard_indices, shift_alarm_indices, tail_score, top_n_indices
from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.rrp_experiments import ScoreRun, load_score_run


KS = (3, 5)
TOP_NS = (100, 300, 500)
Q_UPPERS = (0.90, 0.95)
BIN_PROBS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)


def run_ceres_pilot(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-ceres-smd-pilot"),
    dataset: str = "SMD",
    detector: str = "one_step",
    seeds: Sequence[int] = (0,),
    ks: Sequence[int] = KS,
    top_ns: Sequence[int] = TOP_NS,
    q_uppers: Sequence[float] = Q_UPPERS,
    bin_probs: Sequence[float] = BIN_PROBS,
    calibration_fraction: float = 0.10,
    min_bin_count: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for seed in seeds:
        run = load_score_run(scores_root / dataset / detector / str(seed))
        z_deployment, deployment_calibration = _deployment_normalized(run, calibration_fraction)
        z_oracle, oracle_calibration = _oracle_normalized(run)
        for top_n in top_ns:
            rows.extend(_baseline_rows(dataset, detector, int(seed), run, int(top_n)))
        for k in ks:
            tail = tail_score(run.scores, k=int(k))
            tail_bases: dict[int, np.ndarray] = {}
            for top_n in top_ns:
                tail_base = top_n_indices(tail, int(top_n))
                tail_bases[int(top_n)] = tail_base
                tail_indices = shift_alarm_indices(tail_base, shift=int(k), length=run.scores.size)
                rows.append(
                    _metric_row(
                        dataset=dataset,
                        detector=detector,
                        seed=int(seed),
                        setting="baseline",
                        method="tail_K",
                        score=tail,
                        alarm_indices=tail_indices,
                        labels=run.labels,
                        top_n=int(top_n),
                        k=int(k),
                        q_upper=np.nan,
                        calibration_size=0,
                        tail_jaccard=1.0,
                    )
                )
            for q_upper in q_uppers:
                for setting, z_all, z_cal in (
                    ("deployment", z_deployment, deployment_calibration),
                    ("oracle", z_oracle, oracle_calibration),
                ):
                    env, diagnostic_rows = build_tail_envelopes(
                        z_cal,
                        k=int(k),
                        bin_probs=bin_probs,
                        q_upper=float(q_upper),
                        min_bin_count=int(min_bin_count),
                    )
                    diagnostics.extend(
                        {
                            "dataset": dataset,
                            "detector": detector,
                            "seed": int(seed),
                            "setting": setting,
                            "k": int(k),
                            "q_upper": float(q_upper),
                            "min_bin_count": int(min_bin_count),
                            **row,
                        }
                        for row in diagnostic_rows
                    )
                    for method, score in (
                        ("ceres_lite", ceres_lite_score(z_all, env, k=int(k))),
                        ("ceres_envelope", ceres_envelope_score(z_all, env, k=int(k))),
                    ):
                        for top_n in top_ns:
                            base = top_n_indices(score, int(top_n))
                            shifted = shift_alarm_indices(base, shift=int(k), length=run.scores.size)
                            rows.append(
                                _metric_row(
                                    dataset=dataset,
                                    detector=detector,
                                    seed=int(seed),
                                    setting=setting,
                                    method=method,
                                    score=score,
                                    alarm_indices=shifted,
                                    labels=run.labels,
                                    top_n=int(top_n),
                                    k=int(k),
                                    q_upper=float(q_upper),
                                    calibration_size=_calibration_size(z_cal),
                                    tail_jaccard=jaccard_indices(base, tail_bases[int(top_n)]),
                                )
                            )

    frame = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)
    best = _best_rows(frame)
    gate = _gate_summary(best)
    _write_outputs(output_dir, frame, diag, best, gate)
    return frame, best, gate


def _deployment_normalized(run: ScoreRun, calibration_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    n = run.scores.size
    cal_len = max(1, int(np.ceil(n * float(calibration_fraction))))
    cal_len = min(n, cal_len)
    center, scale = robust_location_scale(run.scores[:cal_len])
    z_all, _ = normalize_scores(run.scores, center=center, scale=scale)
    return z_all, z_all[:cal_len]


def _oracle_normalized(run: ScoreRun) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(run.labels).astype(bool).reshape(-1)
    normal_scores = run.scores[~labels]
    if normal_scores.size == 0:
        return _deployment_normalized(run, 0.10)
    center, scale = robust_location_scale(normal_scores)
    z_all, _ = normalize_scores(run.scores, center=center, scale=scale)
    z_cal = z_all.copy()
    z_cal[labels] = np.nan
    return z_all, z_cal


def _baseline_rows(dataset: str, detector: str, seed: int, run: ScoreRun, top_n: int) -> list[dict[str, object]]:
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
            q_upper=np.nan,
            calibration_size=0,
            tail_jaccard=np.nan,
        )
    ]


def _ceres_rows(
    dataset: str,
    detector: str,
    seed: int,
    setting: str,
    labels: np.ndarray,
    z_all: np.ndarray,
    z_cal: np.ndarray,
    k: int,
    top_n: int,
    q_upper: float,
    bin_probs: Sequence[float],
    min_bin_count: int,
    tail_base: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    env, diagnostics = build_tail_envelopes(
        z_cal,
        k=k,
        bin_probs=bin_probs,
        q_upper=q_upper,
        min_bin_count=min_bin_count,
    )
    rows: list[dict[str, object]] = []
    for method, score in (
        ("ceres_lite", ceres_lite_score(z_all, env, k=k)),
        ("ceres_envelope", ceres_envelope_score(z_all, env, k=k)),
    ):
        base = top_n_indices(score, top_n)
        shifted = shift_alarm_indices(base, shift=k, length=labels.size)
        rows.append(
            _metric_row(
                dataset=dataset,
                detector=detector,
                seed=seed,
                setting=setting,
                method=method,
                score=score,
                alarm_indices=shifted,
                labels=labels,
                top_n=top_n,
                k=k,
                q_upper=q_upper,
                calibration_size=_calibration_size(z_cal),
                tail_jaccard=jaccard_indices(base, tail_base),
            )
        )
    diag_rows = [
        {
            "dataset": dataset,
            "detector": detector,
            "seed": seed,
            "setting": setting,
            "k": k,
            "q_upper": q_upper,
            "min_bin_count": min_bin_count,
            **row,
        }
        for row in diagnostics
    ]
    return rows, diag_rows


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
    q_upper: float,
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
        "seed": seed,
        "setting": setting,
        "method": method,
        "top_n": int(top_n),
        "k": int(k),
        "q_upper": float(q_upper) if np.isfinite(q_upper) else np.nan,
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
    return int(np.logical_and(np.isfinite(arr), arr > 0.0).sum())


def _calibration_size(z_cal: np.ndarray) -> int:
    return int(np.isfinite(np.asarray(z_cal, dtype=np.float64)).sum())


def _best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["family"] = frame["setting"].astype(str) + "_" + frame["method"].astype(str)
    families = [
        "baseline_ewma",
        "baseline_tail_K",
        "deployment_ceres_lite",
        "deployment_ceres_envelope",
        "oracle_ceres_lite",
        "oracle_ceres_envelope",
    ]
    rows: list[pd.Series] = []
    for family in families:
        subset = frame[frame["family"] == family]
        if subset.empty:
            continue
        idx = subset.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).index[0]
        rows.append(subset.loc[idx].copy())
    return pd.DataFrame(rows)


def _gate_summary(best: pd.DataFrame) -> pd.DataFrame:
    baseline = best[best["family"].isin(["baseline_ewma", "baseline_tail_K"])]
    ceres = best[best["family"].isin(["deployment_ceres_lite", "deployment_ceres_envelope"])]
    if baseline.empty or ceres.empty:
        return pd.DataFrame(columns=_gate_columns())

    baseline_row = baseline.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).iloc[0]
    ceres_row = ceres.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).iloc[0]
    raw_f1_pass = bool(ceres_row["raw_f1"] >= baseline_row["raw_f1"] + 0.02)
    recall_pass = bool(ceres_row["event_recall"] >= baseline_row["event_recall"] - 0.03)
    mttd_pass = _mttd_pass(ceres_row["mttd"], baseline_row["mttd"], int(ceres_row["k"]))
    event_pass = bool(ceres_row["predicted_events"] <= baseline_row["predicted_events"] * 1.10)
    jaccard_pass = bool(ceres_row["tail_jaccard"] <= 0.90)
    gate_pass = bool(raw_f1_pass and recall_pass and mttd_pass and event_pass and jaccard_pass)
    return pd.DataFrame(
        [
            {
                "baseline_family": baseline_row["family"],
                "baseline_method": baseline_row["method"],
                "baseline_raw_f1": baseline_row["raw_f1"],
                "ceres_family": ceres_row["family"],
                "ceres_method": ceres_row["method"],
                "ceres_setting": ceres_row["setting"],
                "ceres_raw_f1": ceres_row["raw_f1"],
                "raw_f1_pass": raw_f1_pass,
                "baseline_event_recall": baseline_row["event_recall"],
                "ceres_event_recall": ceres_row["event_recall"],
                "event_recall_pass": recall_pass,
                "baseline_mttd": baseline_row["mttd"],
                "ceres_mttd": ceres_row["mttd"],
                "ceres_k": int(ceres_row["k"]),
                "mttd_pass": mttd_pass,
                "baseline_predicted_events": baseline_row["predicted_events"],
                "ceres_predicted_events": ceres_row["predicted_events"],
                "predicted_events_pass": event_pass,
                "ceres_tail_jaccard": ceres_row["tail_jaccard"],
                "jaccard_pass": jaccard_pass,
                "baseline_top_n": int(baseline_row["top_n"]),
                "ceres_top_n": int(ceres_row["top_n"]),
                "ceres_q_upper": ceres_row["q_upper"],
                "gate_pass": gate_pass,
            }
        ]
    )


def _mttd_pass(ceres_mttd: float, baseline_mttd: float, k: int) -> bool:
    ceres_nan = pd.isna(ceres_mttd)
    baseline_nan = pd.isna(baseline_mttd)
    if ceres_nan and baseline_nan:
        return True
    if ceres_nan:
        return False
    if baseline_nan:
        return True
    return bool(ceres_mttd <= baseline_mttd + k)


def _gate_columns() -> list[str]:
    return [
        "baseline_family",
        "baseline_method",
        "baseline_raw_f1",
        "ceres_family",
        "ceres_method",
        "ceres_setting",
        "ceres_raw_f1",
        "raw_f1_pass",
        "baseline_event_recall",
        "ceres_event_recall",
        "event_recall_pass",
        "baseline_mttd",
        "ceres_mttd",
        "ceres_k",
        "mttd_pass",
        "baseline_predicted_events",
        "ceres_predicted_events",
        "predicted_events_pass",
        "ceres_tail_jaccard",
        "jaccard_pass",
        "baseline_top_n",
        "ceres_top_n",
        "ceres_q_upper",
        "gate_pass",
    ]


def _write_outputs(
    output_dir: Path,
    frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "ceres_all_rows.csv", index=False)
    diagnostics.to_csv(tables_dir / "ceres_bin_diagnostics.csv", index=False)
    best.to_csv(tables_dir / "ceres_best_by_family.csv", index=False)
    gate.to_csv(tables_dir / "ceres_gate_report.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CERES SMD pilot.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-ceres-smd-pilot"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detector", default="one_step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(TOP_NS))
    parser.add_argument("--q-uppers", nargs="+", type=float, default=list(Q_UPPERS))
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-bin-count", type=int, default=30)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_ceres_pilot(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detector=args.detector,
        seeds=tuple(args.seeds),
        ks=tuple(args.ks),
        top_ns=tuple(args.top_ns),
        q_uppers=tuple(args.q_uppers),
        calibration_fraction=float(args.calibration_fraction),
        min_bin_count=int(args.min_bin_count),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
