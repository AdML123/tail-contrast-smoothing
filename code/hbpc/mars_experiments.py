from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.mars import (
    ewma_score,
    jaccard_indices,
    mars_abs_score,
    mars_rel_score,
    shift_alarm_indices,
    tail_score,
    top_n_indices,
)
from hbpc.metrics import detection_delay, events_from_binary, f1, point_adjust
from hbpc.rrp_experiments import ScoreRun, load_score_run


HORIZONS = (3, 5)
TOP_NS = (100, 300, 500)
ALPHAS = (0.5, 1.0, 2.0)


def run_mars_pilot(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-mars-smd-pilot"),
    dataset: str = "SMD",
    detector: str = "one_step",
    seeds: Sequence[int] = (0,),
    horizons: Sequence[int] = HORIZONS,
    top_ns: Sequence[int] = TOP_NS,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        run = load_score_run(scores_root / dataset / detector / str(seed))
        for top_n in top_ns:
            rows.extend(_baseline_rows(dataset, detector, int(seed), run, int(top_n)))
            for horizon in horizons:
                rows.extend(_mars_rows(dataset, detector, int(seed), run, int(horizon), int(top_n), alphas))

    frame = pd.DataFrame(rows)
    best = _best_rows(frame)
    gate = _gate_summary(best)
    _write_outputs(output_dir, frame, best, gate)
    return frame, best, gate


def _baseline_rows(dataset: str, detector: str, seed: int, run: ScoreRun, top_n: int) -> list[dict[str, object]]:
    specs = [
        ("raw", run.scores, 0, np.nan),
        ("ewma", ewma_score(run.scores, alpha=0.3), 0, np.nan),
    ]
    rows = [
        _metric_row(
            dataset=dataset,
            detector=detector,
            seed=seed,
            method=method,
            score=score,
            alarm_indices=top_n_indices(score, top_n),
            labels=run.labels,
            top_n=top_n,
            k=k,
            alpha=alpha,
            tail_jaccard=np.nan,
        )
        for method, score, k, alpha in specs
    ]
    return rows


def _mars_rows(
    dataset: str,
    detector: str,
    seed: int,
    run: ScoreRun,
    k: int,
    top_n: int,
    alphas: Sequence[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tail = tail_score(run.scores, k=k)
    tail_base = top_n_indices(tail, top_n)
    tail_indices = shift_alarm_indices(tail_base, shift=k, length=run.scores.size)
    rows.append(
        _metric_row(
            dataset=dataset,
            detector=detector,
            seed=seed,
            method="tail_K",
            score=tail,
            alarm_indices=tail_indices,
            labels=run.labels,
            top_n=top_n,
            k=k,
            alpha=0.0,
            tail_jaccard=1.0,
        )
    )

    for alpha in alphas:
        for method, score in (
            ("mars_abs", mars_abs_score(run.scores, k=k, alpha=float(alpha))),
            ("mars_rel", mars_rel_score(run.scores, k=k, alpha=float(alpha))),
        ):
            base = top_n_indices(score, top_n)
            shifted = shift_alarm_indices(base, shift=k, length=run.scores.size)
            rows.append(
                _metric_row(
                    dataset=dataset,
                    detector=detector,
                    seed=seed,
                    method=method,
                    score=score,
                    alarm_indices=shifted,
                    labels=run.labels,
                    top_n=top_n,
                    k=k,
                    alpha=float(alpha),
                    tail_jaccard=jaccard_indices(base, tail_base),
                )
            )
    return rows


def _metric_row(
    dataset: str,
    detector: str,
    seed: int,
    method: str,
    score: np.ndarray,
    alarm_indices: np.ndarray,
    labels: np.ndarray,
    top_n: int,
    k: int,
    alpha: float,
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
        "method": method,
        "top_n": int(top_n),
        "k": int(k),
        "alpha": float(alpha) if np.isfinite(alpha) else np.nan,
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


def _best_rows(frame: pd.DataFrame) -> pd.DataFrame:
    families = {
        "raw": frame[frame["method"] == "raw"],
        "ewma": frame[frame["method"] == "ewma"],
        "tail_K": frame[frame["method"] == "tail_K"],
        "mars_abs": frame[frame["method"] == "mars_abs"],
        "mars_rel": frame[frame["method"] == "mars_rel"],
    }
    rows: list[pd.Series] = []
    for family, subset in families.items():
        if subset.empty:
            continue
        idx = subset.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).index[0]
        row = subset.loc[idx].copy()
        row["family"] = family
        rows.append(row)
    return pd.DataFrame(rows)


def _gate_summary(best: pd.DataFrame) -> pd.DataFrame:
    baseline = best[best["family"].isin(["ewma", "tail_K"])]
    mars = best[best["family"].isin(["mars_abs", "mars_rel"])]
    if baseline.empty or mars.empty:
        return pd.DataFrame(columns=_gate_columns())

    baseline_row = baseline.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).iloc[0]
    mars_row = mars.sort_values(["raw_f1", "event_recall", "pa_f1"], ascending=[False, False, False]).iloc[0]
    raw_f1_pass = bool(mars_row["raw_f1"] >= baseline_row["raw_f1"] + 0.02)
    recall_pass = bool(mars_row["event_recall"] >= baseline_row["event_recall"] - 0.03)
    mttd_pass = _mttd_pass(mars_row["mttd"], baseline_row["mttd"], int(mars_row["k"]))
    event_pass = bool(mars_row["predicted_events"] <= baseline_row["predicted_events"] * 1.10)
    jaccard_pass = bool(mars_row["tail_jaccard"] <= 0.90)
    gate_pass = bool(raw_f1_pass and recall_pass and mttd_pass and event_pass and jaccard_pass)
    return pd.DataFrame(
        [
            {
                "baseline_family": baseline_row["family"],
                "baseline_method": baseline_row["method"],
                "baseline_raw_f1": baseline_row["raw_f1"],
                "mars_family": mars_row["family"],
                "mars_method": mars_row["method"],
                "mars_raw_f1": mars_row["raw_f1"],
                "raw_f1_pass": raw_f1_pass,
                "baseline_event_recall": baseline_row["event_recall"],
                "mars_event_recall": mars_row["event_recall"],
                "event_recall_pass": recall_pass,
                "baseline_mttd": baseline_row["mttd"],
                "mars_mttd": mars_row["mttd"],
                "mars_k": int(mars_row["k"]),
                "mttd_pass": mttd_pass,
                "baseline_predicted_events": baseline_row["predicted_events"],
                "mars_predicted_events": mars_row["predicted_events"],
                "predicted_events_pass": event_pass,
                "mars_tail_jaccard": mars_row["tail_jaccard"],
                "jaccard_pass": jaccard_pass,
                "baseline_top_n": int(baseline_row["top_n"]),
                "mars_top_n": int(mars_row["top_n"]),
                "mars_alpha": mars_row["alpha"],
                "gate_pass": gate_pass,
            }
        ]
    )


def _mttd_pass(mars_mttd: float, baseline_mttd: float, k: int) -> bool:
    mars_nan = pd.isna(mars_mttd)
    baseline_nan = pd.isna(baseline_mttd)
    if mars_nan and baseline_nan:
        return True
    if mars_nan:
        return False
    if baseline_nan:
        return True
    return bool(mars_mttd <= baseline_mttd + k)


def _gate_columns() -> list[str]:
    return [
        "baseline_family",
        "baseline_method",
        "baseline_raw_f1",
        "mars_family",
        "mars_method",
        "mars_raw_f1",
        "raw_f1_pass",
        "baseline_event_recall",
        "mars_event_recall",
        "event_recall_pass",
        "baseline_mttd",
        "mars_mttd",
        "mars_k",
        "mttd_pass",
        "baseline_predicted_events",
        "mars_predicted_events",
        "predicted_events_pass",
        "mars_tail_jaccard",
        "jaccard_pass",
        "baseline_top_n",
        "mars_top_n",
        "mars_alpha",
        "gate_pass",
    ]


def _write_outputs(output_dir: Path, frame: pd.DataFrame, best: pd.DataFrame, gate: pd.DataFrame) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "mars_rows.csv", index=False)
    best.to_csv(tables_dir / "mars_best.csv", index=False)
    gate.to_csv(tables_dir / "mars_gate.csv", index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the MARS SMD pilot.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-mars-smd-pilot"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detector", default="one_step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(TOP_NS))
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_mars_pilot(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detector=args.detector,
        seeds=tuple(args.seeds),
        horizons=tuple(args.horizons),
        top_ns=tuple(args.top_ns),
        alphas=tuple(args.alphas),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
