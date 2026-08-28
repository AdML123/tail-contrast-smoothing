from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.rpe import median_smooth, quantile_peak_events, robust_zscore, top_n_peak_events
from hbpc.thresholds import threshold_scores


SMD_DETECTORS = ("one_step", "multi_mean_raw", "multi_mean_norm_rms_clip")
SMOOTH_WINDOWS = (3, 5)
REFRACTORIES = (10, 25, 50, 100)
TOP_NS = (100, 300, 500, 1000)
QUANTILES = (0.99, 0.995, 0.999)


@dataclass(frozen=True)
class ScoreRun:
    scores: np.ndarray
    labels: np.ndarray


def load_score_run(path: Path | str) -> ScoreRun:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path / "scores.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        if "scores" not in data or "labels" not in data:
            raise KeyError(f"{npz_path} must contain scores and labels")
        scores = np.asarray(data["scores"], dtype=float).reshape(-1)
        labels = np.asarray(data["labels"]).astype(bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    return ScoreRun(scores=scores, labels=labels)


def event_recall(pred: np.ndarray, labels: np.ndarray) -> float:
    pred_arr = np.asarray(pred).astype(bool).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    if pred_arr.shape != labels_arr.shape:
        raise ValueError("pred and labels must have the same shape")
    events = events_from_binary(labels_arr)
    if not events:
        return 0.0
    hits = sum(bool(pred_arr[start : stop + 1].any()) for start, stop in events)
    return float(hits / len(events))


def run_rpe_gate(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-rpe-smd-gate"),
    dataset: str = "SMD",
    detectors: Sequence[str] = SMD_DETECTORS,
    seeds: Sequence[int] = (0,),
    smooth_windows: Sequence[int] = SMOOTH_WINDOWS,
    refractories: Sequence[int] = REFRACTORIES,
    top_ns: Sequence[int] = TOP_NS,
    quantiles: Sequence[float] = QUANTILES,
    calibration_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    for detector in detectors:
        for seed in seeds:
            run = load_score_run(scores_root / dataset / detector / str(seed))
            reference = _calibration_reference(run.scores, calibration_fraction)

            for quantile in quantiles:
                threshold = _finite_quantile(run.scores, quantile)
                pred = threshold_scores(run.scores, threshold)
                rows.append(
                    _metric_row(
                        dataset=dataset,
                        detector=detector,
                        seed=seed,
                        method="quantile_baseline",
                        pred=pred,
                        labels=run.labels,
                        quantile=quantile,
                        threshold=threshold,
                    )
                )

            z = robust_zscore(run.scores, reference)
            for smooth_window in smooth_windows:
                smoothed = median_smooth(z, window=smooth_window)
                for refractory in refractories:
                    for top_n in top_ns:
                        pred = top_n_peak_events(smoothed, n=top_n, refractory=refractory)
                        rows.append(
                            _metric_row(
                                dataset=dataset,
                                detector=detector,
                                seed=seed,
                                method="rpe_top_n",
                                pred=pred,
                                labels=run.labels,
                                smooth_window=smooth_window,
                                refractory=refractory,
                                top_n=top_n,
                            )
                        )
                    for quantile in quantiles:
                        pred = quantile_peak_events(smoothed, quantile=quantile, refractory=refractory)
                        rows.append(
                            _metric_row(
                                dataset=dataset,
                                detector=detector,
                                seed=seed,
                                method="rpe_peak_quantile",
                                pred=pred,
                                labels=run.labels,
                                smooth_window=smooth_window,
                                refractory=refractory,
                                quantile=quantile,
                            )
                        )

    frame = pd.DataFrame(rows)
    gate = _write_outputs(output_dir, frame)
    return frame, gate


def _metric_row(
    dataset: str,
    detector: str,
    seed: int,
    method: str,
    pred: np.ndarray,
    labels: np.ndarray,
    quantile: float | None = None,
    threshold: float | None = None,
    smooth_window: int | None = None,
    refractory: int | None = None,
    top_n: int | None = None,
) -> dict[str, object]:
    pred_arr = np.asarray(pred).astype(int).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    pred_events = events_from_binary(pred_arr)
    label_events = events_from_binary(labels_arr)
    adjusted = point_adjust(pred_arr, labels_arr)
    return {
        "dataset": dataset,
        "detector": detector,
        "seed": seed,
        "method": method,
        "quantile": np.nan if quantile is None else float(quantile),
        "threshold": np.nan if threshold is None else float(threshold),
        "smooth_window": -1 if smooth_window is None else int(smooth_window),
        "refractory": -1 if refractory is None else int(refractory),
        "top_n": -1 if top_n is None else int(top_n),
        "raw_f1": f1(pred_arr, labels_arr),
        "pa_f1": f1(adjusted, labels_arr),
        "event_recall": event_recall(pred_arr, labels_arr),
        "delay": detection_delay(pred_arr, labels_arr),
        "predicted_points": int(pred_arr.sum()),
        "predicted_events": len(pred_events),
        "label_points": int(labels_arr.sum()),
        "label_events": len(label_events),
    }


def _write_outputs(output_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(metrics_dir / "rpe_all_rows.csv", index=False)

    best = _best_by_method(frame)
    best.to_csv(tables_dir / "rpe_best_by_method.csv", index=False)

    gate = _gate_summary(best)
    gate.to_csv(tables_dir / "rpe_gate_summary.csv", index=False)
    return gate


def _best_by_method(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    groups = {
        "baseline": frame[frame["method"] == "quantile_baseline"],
        "rpe_top_n": frame[frame["method"] == "rpe_top_n"],
        "rpe_peak_quantile": frame[frame["method"] == "rpe_peak_quantile"],
    }
    for family, subset in groups.items():
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
    baseline = best[best["family"] == "baseline"]
    rpe = best[best["family"] == "rpe_top_n"]
    if baseline.empty or rpe.empty:
        return pd.DataFrame(columns=_gate_columns())

    baseline_row = baseline.iloc[0]
    rpe_row = rpe.iloc[0]
    raw_f1_pass = bool(rpe_row["raw_f1"] >= baseline_row["raw_f1"] + 0.05)
    recall_pass = bool(rpe_row["event_recall"] >= baseline_row["event_recall"])
    delay_pass = _delay_not_worse(rpe_row["delay"], baseline_row["delay"])
    point_pass = bool(rpe_row["predicted_points"] < baseline_row["predicted_points"])
    event_pass = bool(rpe_row["predicted_events"] < baseline_row["predicted_events"])
    gate_pass = bool(raw_f1_pass and recall_pass and delay_pass and point_pass and event_pass)
    return pd.DataFrame(
        [
            {
                "baseline_raw_f1": baseline_row["raw_f1"],
                "rpe_raw_f1": rpe_row["raw_f1"],
                "raw_f1_pass": raw_f1_pass,
                "baseline_event_recall": baseline_row["event_recall"],
                "rpe_event_recall": rpe_row["event_recall"],
                "event_recall_pass": recall_pass,
                "baseline_delay": baseline_row["delay"],
                "rpe_delay": rpe_row["delay"],
                "delay_pass": delay_pass,
                "baseline_predicted_points": baseline_row["predicted_points"],
                "rpe_predicted_points": rpe_row["predicted_points"],
                "predicted_points_pass": point_pass,
                "baseline_predicted_events": baseline_row["predicted_events"],
                "rpe_predicted_events": rpe_row["predicted_events"],
                "predicted_events_pass": event_pass,
                "gate_pass": gate_pass,
            }
        ]
    )


def _gate_columns() -> list[str]:
    return [
        "baseline_raw_f1",
        "rpe_raw_f1",
        "raw_f1_pass",
        "baseline_event_recall",
        "rpe_event_recall",
        "event_recall_pass",
        "baseline_delay",
        "rpe_delay",
        "delay_pass",
        "baseline_predicted_points",
        "rpe_predicted_points",
        "predicted_points_pass",
        "baseline_predicted_events",
        "rpe_predicted_events",
        "predicted_events_pass",
        "gate_pass",
    ]


def _delay_not_worse(rpe_delay: float, baseline_delay: float) -> bool:
    rpe_nan = pd.isna(rpe_delay)
    baseline_nan = pd.isna(baseline_delay)
    if rpe_nan and baseline_nan:
        return True
    if rpe_nan:
        return False
    if baseline_nan:
        return True
    return bool(rpe_delay <= 1.10 * baseline_delay)


def _calibration_reference(scores: np.ndarray, calibration_fraction: float) -> np.ndarray:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    split = int(arr.size * calibration_fraction)
    split = min(max(split, 1), arr.size)
    return arr[:split]


def _finite_quantile(scores: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise ValueError("scores must contain at least one finite value")
    return float(np.quantile(valid, quantile))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RPE SMD pilot gate.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-rpe-smd-gate"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detectors", nargs="+", default=list(SMD_DETECTORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--smooth-windows", nargs="+", type=int, default=list(SMOOTH_WINDOWS))
    parser.add_argument("--refractories", nargs="+", type=int, default=list(REFRACTORIES))
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(TOP_NS))
    parser.add_argument("--quantiles", nargs="+", type=float, default=list(QUANTILES))
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_rpe_gate(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detectors=tuple(args.detectors),
        seeds=tuple(args.seeds),
        smooth_windows=tuple(args.smooth_windows),
        refractories=tuple(args.refractories),
        top_ns=tuple(args.top_ns),
        quantiles=tuple(args.quantiles),
        calibration_fraction=args.calibration_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
