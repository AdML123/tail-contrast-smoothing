from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.rrp import (
    candidate_count,
    cusum_scores,
    delayed_top_n_alarms,
    ewma_scores,
    peak_gated_tail_scores,
    peak_tail_scores,
    tail_scores,
    top_n_alarms,
)
from hbpc.rrp_experiments import ScoreRun, load_score_run


HORIZONS = (3, 5)
TOP_NS = (100, 300, 500, 1000)
PEAK_QUANTILES = (0.95, 0.98, 0.99)


def run_rrp_detector_pilot(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-rrp-detector-pilot"),
    dataset: str = "SMD",
    detector: str = "one_step",
    seeds: Sequence[int] = (0,),
    horizons: Sequence[int] = HORIZONS,
    top_ns: Sequence[int] = TOP_NS,
    peak_quantiles: Sequence[float] = PEAK_QUANTILES,
    calibration_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        run = load_score_run(scores_root / dataset / detector / str(seed))
        reference = _calibration_reference(run.scores, calibration_fraction)
        for top_n in top_ns:
            rows.extend(_baseline_rows(dataset, detector, int(seed), run, reference, int(top_n)))
            for horizon in horizons:
                rows.extend(
                    _rrp_rows(
                        dataset=dataset,
                        detector=detector,
                        seed=int(seed),
                        run=run,
                        k=int(horizon),
                        top_n=int(top_n),
                        peak_quantiles=peak_quantiles,
                    )
                )

    frame = pd.DataFrame(rows)
    best = _best_rows(frame)
    gate = _gate_summary(best)
    _write_outputs(output_dir, frame, best, gate)
    return frame, best, gate


def _baseline_rows(
    dataset: str,
    detector: str,
    seed: int,
    run: ScoreRun,
    reference: np.ndarray,
    top_n: int,
) -> list[dict[str, object]]:
    specs = [
        ("raw", run.scores),
        ("ewma", ewma_scores(run.scores, alpha=0.3)),
        ("cusum", cusum_scores(run.scores, reference=reference, drift=0.5)),
    ]
    return [
        _metric_row(
            dataset=dataset,
            detector=detector,
            seed=seed,
            method=method,
            score=score,
            pred=top_n_alarms(score, top_n),
            labels=run.labels,
            top_n=top_n,
        )
        for method, score in specs
    ]


def _rrp_rows(
    dataset: str,
    detector: str,
    seed: int,
    run: ScoreRun,
    k: int,
    top_n: int,
    peak_quantiles: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, score in (
        ("tail_K", tail_scores(run.scores, k=k)),
        ("peak_tail_K", peak_tail_scores(run.scores, k=k)),
    ):
        pred = delayed_top_n_alarms(score, k=k, top_n=top_n, length=run.scores.size)
        rows.append(
            _metric_row(
                dataset=dataset,
                detector=detector,
                seed=seed,
                method=method,
                score=score,
                pred=pred,
                labels=run.labels,
                top_n=top_n,
                k=k,
            )
        )

    for quantile in peak_quantiles:
        score = peak_gated_tail_scores(run.scores, k=k, peak_quantile=float(quantile))
        pred = delayed_top_n_alarms(score, k=k, top_n=top_n, length=run.scores.size)
        rows.append(
            _metric_row(
                dataset=dataset,
                detector=detector,
                seed=seed,
                method="peak_gated_tail_K",
                score=score,
                pred=pred,
                labels=run.labels,
                top_n=top_n,
                k=k,
                peak_quantile=float(quantile),
            )
        )
    return rows


def _metric_row(
    dataset: str,
    detector: str,
    seed: int,
    method: str,
    score: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    top_n: int,
    k: int | None = None,
    peak_quantile: float | None = None,
) -> dict[str, object]:
    pred_arr = np.asarray(pred).astype(int).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    adjusted = point_adjust(pred_arr, labels_arr)
    pred_events = events_from_binary(pred_arr)
    label_events = events_from_binary(labels_arr)
    return {
        "dataset": dataset,
        "detector": detector,
        "seed": seed,
        "method": method,
        "top_n": int(top_n),
        "k": -1 if k is None else int(k),
        "peak_quantile": np.nan if peak_quantile is None else float(peak_quantile),
        "raw_f1": f1(pred_arr, labels_arr),
        "pa_f1": f1(adjusted, labels_arr),
        "event_recall": _event_recall(pred_arr, labels_arr),
        "mttd": detection_delay(pred_arr, labels_arr),
        "candidate_count": candidate_count(score),
        "predicted_points": int(top_n),
        "predicted_points_actual": int(pred_arr.sum()),
        "predicted_events": len(pred_events),
        "label_points": int(labels_arr.sum()),
        "label_events": len(label_events),
    }


def _event_recall(pred: np.ndarray, labels: np.ndarray) -> float:
    pred_arr = np.asarray(pred).astype(bool).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    if pred_arr.shape != labels_arr.shape:
        raise ValueError("pred and labels must have the same shape")
    events = events_from_binary(labels_arr)
    if not events:
        return 0.0
    hits = sum(bool(pred_arr[start : stop + 1].any()) for start, stop in events)
    return float(hits / len(events))


def _best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    families = {
        "baseline_raw": frame[frame["method"] == "raw"],
        "ewma": frame[frame["method"] == "ewma"],
        "cusum": frame[frame["method"] == "cusum"],
        "tail_K": frame[frame["method"] == "tail_K"],
        "peak_tail_K": frame[frame["method"] == "peak_tail_K"],
        "peak_gated_tail_K": frame[frame["method"] == "peak_gated_tail_K"],
    }
    rows: list[pd.Series] = []
    for family, subset in families.items():
        if subset.empty:
            continue
        idx = subset.sort_values(
            ["raw_f1", "event_recall", "pa_f1"],
            ascending=[False, False, False],
        ).index[0]
        row = subset.loc[idx].copy()
        row["family"] = family
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_summary(best: pd.DataFrame) -> pd.DataFrame:
    baseline = best[best["family"] == "baseline_raw"]
    rrp = best[best["family"] == "peak_gated_tail_K"]
    if baseline.empty or rrp.empty:
        return pd.DataFrame(columns=_gate_columns())

    baseline_row = baseline.iloc[0]
    rrp_row = rrp.iloc[0]
    raw_f1_pass = bool(rrp_row["raw_f1"] >= baseline_row["raw_f1"] + 0.05)
    recall_pass = bool(rrp_row["event_recall"] >= baseline_row["event_recall"])
    mttd_pass = _mttd_pass(rrp_row["mttd"], baseline_row["mttd"], int(rrp_row["k"]))
    event_pass = bool(rrp_row["predicted_events"] <= baseline_row["predicted_events"])
    gate_pass = bool(raw_f1_pass and recall_pass and mttd_pass and event_pass)
    return pd.DataFrame(
        [
            {
                "baseline_raw_f1": baseline_row["raw_f1"],
                "rrp_raw_f1": rrp_row["raw_f1"],
                "raw_f1_pass": raw_f1_pass,
                "baseline_event_recall": baseline_row["event_recall"],
                "rrp_event_recall": rrp_row["event_recall"],
                "event_recall_pass": recall_pass,
                "baseline_mttd": baseline_row["mttd"],
                "rrp_mttd": rrp_row["mttd"],
                "rrp_k": int(rrp_row["k"]),
                "mttd_pass": mttd_pass,
                "baseline_predicted_events": baseline_row["predicted_events"],
                "rrp_predicted_events": rrp_row["predicted_events"],
                "predicted_events_pass": event_pass,
                "baseline_top_n": int(baseline_row["top_n"]),
                "rrp_top_n": int(rrp_row["top_n"]),
                "rrp_peak_quantile": rrp_row["peak_quantile"],
                "rrp_candidate_count": int(rrp_row["candidate_count"]),
                "rrp_predicted_points_actual": int(rrp_row["predicted_points_actual"]),
                "gate_pass": gate_pass,
            }
        ]
    )


def _mttd_pass(rrp_mttd: float, baseline_mttd: float, k: int) -> bool:
    rrp_nan = pd.isna(rrp_mttd)
    baseline_nan = pd.isna(baseline_mttd)
    if rrp_nan and baseline_nan:
        return True
    if rrp_nan:
        return False
    if baseline_nan:
        return True
    return bool(rrp_mttd <= baseline_mttd + k)


def _gate_columns() -> list[str]:
    return [
        "baseline_raw_f1",
        "rrp_raw_f1",
        "raw_f1_pass",
        "baseline_event_recall",
        "rrp_event_recall",
        "event_recall_pass",
        "baseline_mttd",
        "rrp_mttd",
        "rrp_k",
        "mttd_pass",
        "baseline_predicted_events",
        "rrp_predicted_events",
        "predicted_events_pass",
        "baseline_top_n",
        "rrp_top_n",
        "rrp_peak_quantile",
        "rrp_candidate_count",
        "rrp_predicted_points_actual",
        "gate_pass",
    ]


def _write_outputs(output_dir: Path, frame: pd.DataFrame, best: pd.DataFrame, gate: pd.DataFrame) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "rrp_detector_rows.csv", index=False)
    best.to_csv(tables_dir / "rrp_detector_best.csv", index=False)
    gate.to_csv(tables_dir / "rrp_detector_gate.csv", index=False)


def _calibration_reference(scores: np.ndarray, calibration_fraction: float) -> np.ndarray:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    split = min(max(int(arr.size * calibration_fraction), 1), arr.size)
    return arr[:split]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the short-window RRP detector pilot.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-rrp-detector-pilot"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detector", default="one_step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(TOP_NS))
    parser.add_argument("--peak-quantiles", nargs="+", type=float, default=list(PEAK_QUANTILES))
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_rrp_detector_pilot(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detector=args.detector,
        seeds=tuple(args.seeds),
        horizons=tuple(args.horizons),
        top_ns=tuple(args.top_ns),
        peak_quantiles=tuple(args.peak_quantiles),
        calibration_fraction=args.calibration_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
