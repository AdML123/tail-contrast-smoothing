from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from hbpc.calibration import (
    CalibrationResult,
    eventize,
    feasible_median,
    hn_lowest,
    scan_thresholds,
    stable,
    stable_u,
)
from hbpc.event_metrics import (
    budget_violation_rate,
    miss_rate,
    mttd,
    threshold_fold_cv,
    wr_off,
    wr_total,
)


STAGE1_DETECTORS = ("one_step", "multi_mean_raw", "multi_mean_norm_rms_clip")
STAGE1_BUDGETS = (0.1, 0.5, 1.0, 2.0)
STAGE1_LAMBDAS = (0.1, 0.3, 0.5)


@dataclass(frozen=True)
class ScoreRun:
    scores: np.ndarray
    labels: np.ndarray
    nominal_scores: np.ndarray | None = None
    nominal_source: str | None = None


@dataclass(frozen=True)
class CalibrationStreams:
    calibration_scores: np.ndarray
    eval_scores: np.ndarray
    eval_labels: np.ndarray
    calibration_source: str


def load_score_run(path: Path | str) -> ScoreRun:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path / "scores.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path) as data:
        if "scores" not in data or "labels" not in data:
            raise KeyError(f"{npz_path} must contain scores and labels arrays")
        scores = np.asarray(data["scores"], dtype=float).reshape(-1)
        labels = np.asarray(data["labels"]).astype(bool).reshape(-1)
        nominal_scores = None
        nominal_source = None
        for key in ("train_scores", "nominal_scores", "calibration_scores"):
            if key in data:
                nominal_scores = np.asarray(data[key], dtype=float).reshape(-1)
                nominal_source = key
                break
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    return ScoreRun(
        scores=scores,
        labels=labels,
        nominal_scores=nominal_scores,
        nominal_source=nominal_source,
    )


def select_calibration_streams(
    run: ScoreRun,
    calibration_fraction: float = 0.1,
) -> CalibrationStreams:
    if run.nominal_scores is not None:
        if run.nominal_scores.size == 0:
            raise ValueError("nominal_scores must not be empty")
        return CalibrationStreams(
            calibration_scores=np.asarray(run.nominal_scores, dtype=float).reshape(-1),
            eval_scores=np.asarray(run.scores, dtype=float).reshape(-1),
            eval_labels=np.asarray(run.labels).astype(bool).reshape(-1),
            calibration_source=run.nominal_source or "nominal_scores",
        )

    calibration_scores, eval_scores, eval_labels = split_calibration_eval(
        run.scores,
        run.labels,
        calibration_fraction=calibration_fraction,
    )
    return CalibrationStreams(
        calibration_scores=calibration_scores,
        eval_scores=eval_scores,
        eval_labels=eval_labels,
        calibration_source="test_front_fraction",
    )


def split_calibration_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    calibration_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    scores_arr = np.asarray(scores, dtype=float).reshape(-1)
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError("scores and labels must have the same shape")
    if scores_arr.size < 2:
        raise ValueError("scores must contain at least two points")

    split = int(scores_arr.size * calibration_fraction)
    split = min(max(split, 1), scores_arr.size - 1)
    return scores_arr[:split], scores_arr[split:], labels_arr[split:]


def contiguous_folds(values: np.ndarray, n_folds: int = 5) -> list[np.ndarray]:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    arr = np.asarray(values)
    if arr.size < n_folds:
        raise ValueError("stream length must be at least n_folds")
    return [np.asarray(fold) for fold in np.array_split(arr, n_folds)]


def auto_event_budgets(
    fold_lengths: Sequence[int],
    fs: float = 1.0,
    min_events_per_fold: float = 5.0,
    multipliers: Sequence[float] = (1.0, 2.0, 4.0),
) -> tuple[float, ...]:
    if fs <= 0:
        raise ValueError("fs must be positive")
    if min_events_per_fold <= 0:
        raise ValueError("min_events_per_fold must be positive")
    if not multipliers:
        raise ValueError("multipliers must not be empty")
    lengths = np.asarray(fold_lengths, dtype=float).reshape(-1)
    if lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("fold_lengths must contain positive lengths")
    min_fold_hours = float(np.min(lengths) / (3600.0 * fs))
    base_budget = float(math.ceil(min_events_per_fold / min_fold_hours))
    budgets = sorted({base_budget * float(multiplier) for multiplier in multipliers})
    return tuple(float(budget) for budget in budgets)


def run_stage1_gate(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-stable-u"),
    dataset: str = "SMD",
    detectors: Sequence[str] = STAGE1_DETECTORS,
    seeds: Sequence[int] = (0,),
    budgets: Sequence[float] = STAGE1_BUDGETS,
    lambdas: Sequence[float] = STAGE1_LAMBDAS,
    main_lambda: float = 0.3,
    k: int = 1000,
    gamma: float = 1.0,
    fs: float = 1.0,
    min_gap: int = 0,
    n_folds: int = 5,
    calibration_fraction: float = 0.1,
    auto_budgets: bool = False,
    budget_multipliers: Sequence[float] = (1.0, 2.0, 4.0),
    min_events_per_fold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    stable_lambdas = _stable_lambda_schedule(lambdas, main_lambda)

    for detector in detectors:
        for seed in seeds:
            run = load_score_run(scores_root / dataset / detector / str(seed))
            streams = select_calibration_streams(
                run,
                calibration_fraction=calibration_fraction,
            )
            folds = contiguous_folds(streams.calibration_scores, n_folds=n_folds)
            total_folds = len(seeds) * n_folds
            run_budgets = (
                auto_event_budgets(
                    [len(fold) for fold in folds],
                    fs=fs,
                    min_events_per_fold=min_events_per_fold,
                    multipliers=budget_multipliers,
                )
                if auto_budgets
                else tuple(float(budget) for budget in budgets)
            )

            for budget in run_budgets:
                for fold_idx, fold_scores in enumerate(folds):
                    fold_hours = _fold_hours(fold_scores, fs)
                    budget_events = float(budget) * fold_hours
                    scan = scan_thresholds(
                        fold_scores,
                        budget=float(budget),
                        gamma=gamma,
                        fs=fs,
                        k=k,
                        min_gap=min_gap,
                    )
                    fold_key = _fold_key(seed, fold_idx, total_folds)
                    if budget_events < min_events_per_fold:
                        for calibrator_name, lambda_value in _calibrator_specs(stable_lambdas):
                            row = _infeasible_budget_row(
                                dataset=dataset,
                                detector=detector,
                                seed=seed,
                                budget=float(budget),
                                fold=fold_idx,
                                fold_key=fold_key,
                                calibrator_name=calibrator_name,
                                lambda_value=lambda_value,
                                calibration_source=streams.calibration_source,
                                fold_hours=fold_hours,
                                budget_events=budget_events,
                                min_events_per_fold=min_events_per_fold,
                            )
                            rows.append(row)
                            _write_raw_json(output_dir, row, None)
                        continue

                    calibrations = [
                        (hn_lowest(scan), None, "hn_lowest"),
                        (feasible_median(scan), None, "feasible_median"),
                        (stable(scan, budget=float(budget), gamma=gamma), None, "stable"),
                    ]
                    for lambda_value, calibrator_name in stable_lambdas:
                        result = stable_u(
                            scan,
                            budget=float(budget),
                            gamma=gamma,
                            lambda_=lambda_value,
                        )
                        calibrations.append((result, lambda_value, calibrator_name))

                    for result, lambda_value, calibrator_name in calibrations:
                        row = _evaluate_calibration(
                            dataset=dataset,
                            detector=detector,
                            seed=seed,
                            budget=float(budget),
                            fold=fold_idx,
                            fold_key=fold_key,
                            calibrator_name=calibrator_name,
                            lambda_value=lambda_value,
                            result=result,
                            eval_scores=streams.eval_scores,
                            eval_labels=streams.eval_labels,
                            fs=fs,
                            min_gap=min_gap,
                            calibration_source=streams.calibration_source,
                            fold_hours=fold_hours,
                            budget_events=budget_events,
                            min_events_per_fold=min_events_per_fold,
                        )
                        rows.append(row)
                        _write_raw_json(output_dir, row, result)

    frame = pd.DataFrame(rows)
    frame = _attach_fold_stability(frame)
    _write_metric_outputs(output_dir, frame)
    gate = _write_gate_outputs(output_dir, frame, main_lambda=main_lambda)
    return frame, gate


def _evaluate_calibration(
    dataset: str,
    detector: str,
    seed: int,
    budget: float,
    fold: int,
    fold_key: str,
    calibrator_name: str,
    lambda_value: float | None,
    result: CalibrationResult,
    eval_scores: np.ndarray,
    eval_labels: np.ndarray,
    fs: float,
    min_gap: int,
    calibration_source: str,
    fold_hours: float,
    budget_events: float,
    min_events_per_fold: float,
) -> dict[str, object]:
    pred = np.where(np.isfinite(eval_scores), eval_scores > result.tau, False)
    alarms = eventize(pred, min_gap=min_gap)
    attacks = eventize(eval_labels, min_gap=min_gap)
    row = {
        "dataset": dataset,
        "detector": detector,
        "seed": seed,
        "budget": budget,
        "fold": fold,
        "fold_key": fold_key,
        "calibrator": calibrator_name,
        "lambda": -1.0 if lambda_value is None else float(lambda_value),
        "calibration_source": calibration_source,
        "fold_hours": fold_hours,
        "budget_events": budget_events,
        "min_events_per_fold": min_events_per_fold,
        "budget_status": "valid",
        "participates_in_gate": True,
        "tau": result.tau,
        "tau_quantile": result.tau_quantile,
        "fae_nom": result.fae,
        "duty_nom": result.duty,
        "miss_rate": miss_rate(pred, eval_labels, min_gap=min_gap),
        "mttd": mttd(pred, eval_labels, min_gap=min_gap),
        "wr_off": wr_off(pred, eval_labels, budget=budget, fs=fs, min_gap=min_gap),
        "wr_total": wr_total(pred, budget=budget, fs=fs, min_gap=min_gap),
        "predicted_points": int(pred.sum()),
        "predicted_events": len(alarms),
        "label_points": int(np.asarray(eval_labels).astype(bool).sum()),
        "label_events": len(attacks),
        "low_util_warning": bool(result.low_util_warning),
        "plateau_lo": result.plateau_lo,
        "plateau_hi": result.plateau_hi,
        "W_q": result.w_q,
        "M_rel": result.m_rel,
        "U": result.u,
        "median_fae": result.median_fae,
        "delta_rel": result.delta_rel,
        "stability": result.stability,
    }
    row["budget_violation"] = bool(row["wr_off"] > 1.0)
    return row


def _infeasible_budget_row(
    dataset: str,
    detector: str,
    seed: int,
    budget: float,
    fold: int,
    fold_key: str,
    calibrator_name: str,
    lambda_value: float | None,
    calibration_source: str,
    fold_hours: float,
    budget_events: float,
    min_events_per_fold: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "detector": detector,
        "seed": seed,
        "budget": budget,
        "fold": fold,
        "fold_key": fold_key,
        "calibrator": calibrator_name,
        "lambda": -1.0 if lambda_value is None else float(lambda_value),
        "calibration_source": calibration_source,
        "fold_hours": fold_hours,
        "budget_events": budget_events,
        "min_events_per_fold": min_events_per_fold,
        "budget_status": "infeasible_discrete",
        "participates_in_gate": False,
        "tau": np.nan,
        "tau_quantile": np.nan,
        "fae_nom": np.nan,
        "duty_nom": np.nan,
        "miss_rate": np.nan,
        "mttd": np.nan,
        "wr_off": np.nan,
        "wr_total": np.nan,
        "predicted_points": 0,
        "predicted_events": 0,
        "label_points": 0,
        "label_events": 0,
        "low_util_warning": False,
        "plateau_lo": None,
        "plateau_hi": None,
        "W_q": np.nan,
        "M_rel": np.nan,
        "U": np.nan,
        "median_fae": np.nan,
        "delta_rel": np.nan,
        "stability": np.nan,
        "budget_violation": False,
    }


def _attach_fold_stability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    group_cols = ["dataset", "detector", "seed", "budget", "calibrator", "lambda"]
    frame["tau_cv"] = np.nan
    frame["p_cv"] = np.nan
    valid = _participates_mask(frame)
    for _, index in frame[valid].groupby(group_cols, dropna=False).groups.items():
        idx = list(index)
        frame.loc[idx, "tau_cv"] = threshold_fold_cv(frame.loc[idx, "tau"])
        frame.loc[idx, "p_cv"] = threshold_fold_cv(frame.loc[idx, "tau_quantile"])
    return frame


def _write_metric_outputs(output_dir: Path, frame: pd.DataFrame) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(metrics_dir / "stable_u_all_rows.csv", index=False)
    status = (
        frame.groupby(
            ["dataset", "detector", "budget", "calibration_source", "budget_status"],
            as_index=False,
            dropna=False,
        )
        .agg(rows=("budget_status", "size"))
        .sort_values(["dataset", "detector", "budget", "budget_status"])
    )
    status.to_csv(tables_dir / "stage1_budget_status.csv", index=False)

    valid = frame[_participates_mask(frame)].copy()
    if valid.empty:
        summary = pd.DataFrame(
            columns=[
                "calibrator",
                "miss_rate",
                "mttd",
                "wr_off",
                "wr_total",
                "p_cv",
                "tau_cv",
                "budget_violation_rate",
                "low_util_warning_rate",
            ]
        )
    else:
        summary = (
            valid.groupby(["calibrator"], as_index=False)
            .agg(
                miss_rate=("miss_rate", "mean"),
                mttd=("mttd", "mean"),
                wr_off=("wr_off", "mean"),
                wr_total=("wr_total", "mean"),
                p_cv=("p_cv", "mean"),
                tau_cv=("tau_cv", "mean"),
                budget_violation_rate=("wr_off", budget_violation_rate),
                low_util_warning_rate=("low_util_warning", "mean"),
            )
            .sort_values("calibrator")
        )
    summary.to_csv(tables_dir / "stage1_gate_summary.csv", index=False)


def _write_gate_outputs(
    output_dir: Path,
    frame: pd.DataFrame,
    main_lambda: float,
) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    valid = frame[_participates_mask(frame)].copy()
    main = valid[
        (valid["calibrator"].isin(["hn_lowest", "stable_u"]))
        & (
            (valid["calibrator"] != "stable_u")
            | np.isclose(valid["lambda"], float(main_lambda))
        )
    ].copy()
    if main.empty:
        gate = _empty_gate_frame()
        gate.to_csv(tables_dir / "stage1_gate_report.csv", index=False)
        return gate
    means = (
        main.groupby(["dataset", "detector", "budget", "calibrator"], as_index=False)
        .agg(
            miss_rate=("miss_rate", "mean"),
            mttd=("mttd", "mean"),
            wr_off=("wr_off", "mean"),
            p_cv=("p_cv", "mean"),
        )
    )
    hn = means[means["calibrator"] == "hn_lowest"].set_index(["dataset", "detector", "budget"])
    su = means[means["calibrator"] == "stable_u"].set_index(["dataset", "detector", "budget"])

    rows: list[dict[str, object]] = []
    for key in sorted(hn.index.intersection(su.index)):
        hn_row = hn.loc[key]
        su_row = su.loc[key]
        wr_off_pass = _relative_reduction_pass(su_row["wr_off"], hn_row["wr_off"])
        p_cv_pass = _relative_reduction_pass(su_row["p_cv"], hn_row["p_cv"])
        mr_pass = bool(su_row["miss_rate"] <= hn_row["miss_rate"] + 0.05)
        mttd_pass = _mttd_not_worse(su_row["mttd"], hn_row["mttd"])
        rows.append(
            {
                "dataset": key[0],
                "detector": key[1],
                "budget": key[2],
                "hn_wr_off": hn_row["wr_off"],
                "stable_u_wr_off": su_row["wr_off"],
                "wr_off_pass": wr_off_pass,
                "hn_p_cv": hn_row["p_cv"],
                "stable_u_p_cv": su_row["p_cv"],
                "p_cv_pass": p_cv_pass,
                "hn_miss_rate": hn_row["miss_rate"],
                "stable_u_miss_rate": su_row["miss_rate"],
                "miss_rate_pass": mr_pass,
                "hn_mttd": hn_row["mttd"],
                "stable_u_mttd": su_row["mttd"],
                "mttd_pass": mttd_pass,
                "gate_pass": bool(wr_off_pass and p_cv_pass and mr_pass and mttd_pass),
            }
        )

    gate = pd.DataFrame(rows)
    gate.to_csv(tables_dir / "stage1_gate_report.csv", index=False)
    return gate


def _empty_gate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "detector",
            "budget",
            "hn_wr_off",
            "stable_u_wr_off",
            "wr_off_pass",
            "hn_p_cv",
            "stable_u_p_cv",
            "p_cv_pass",
            "hn_miss_rate",
            "stable_u_miss_rate",
            "miss_rate_pass",
            "hn_mttd",
            "stable_u_mttd",
            "mttd_pass",
            "gate_pass",
        ]
    )


def _participates_mask(frame: pd.DataFrame) -> pd.Series:
    if "participates_in_gate" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["participates_in_gate"].fillna(True).astype(bool)


def _relative_reduction_pass(
    stable_value: float,
    baseline_value: float,
    factor: float = 0.8,
) -> bool:
    if pd.isna(stable_value) or pd.isna(baseline_value):
        return False
    if baseline_value <= 0:
        return False
    return bool(stable_value <= factor * baseline_value)


def _write_raw_json(
    output_dir: Path,
    row: dict[str, object],
    result: CalibrationResult | None,
) -> None:
    path = (
        output_dir
        / "raw"
        / str(row["dataset"])
        / str(row["detector"])
        / _budget_key(float(row["budget"]))
        / str(row["fold_key"])
        / f"{row['calibrator']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**row, "calibration_result": None if result is None else asdict(result)}
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _stable_lambda_schedule(
    lambdas: Sequence[float],
    main_lambda: float,
) -> list[tuple[float, str]]:
    unique = sorted({float(value) for value in lambdas} | {float(main_lambda)})
    schedule: list[tuple[float, str]] = []
    for value in unique:
        name = "stable_u" if np.isclose(value, main_lambda) else f"stable_u_l{value:g}"
        schedule.append((value, name))
    return schedule


def _calibrator_specs(
    stable_lambdas: Sequence[tuple[float, str]],
) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = [
        ("hn_lowest", None),
        ("feasible_median", None),
        ("stable", None),
    ]
    specs.extend((name, value) for value, name in stable_lambdas)
    return specs


def _fold_hours(fold_scores: np.ndarray, fs: float) -> float:
    if fs <= 0:
        raise ValueError("fs must be positive")
    return float(len(fold_scores) / (3600.0 * fs))


def _mttd_not_worse(stable_value: float, hn_value: float) -> bool:
    stable_nan = pd.isna(stable_value)
    hn_nan = pd.isna(hn_value)
    if stable_nan and hn_nan:
        return True
    if stable_nan:
        return False
    if hn_nan:
        return True
    return bool(stable_value <= 1.10 * hn_value)


def _fold_key(seed: int, fold: int, total_folds: int) -> str:
    if total_folds <= 5 and seed == 0:
        return str(fold)
    return f"seed{seed}_fold{fold}"


def _budget_key(budget: float) -> str:
    return f"{budget:g}"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the STABLE-U SMD calibration gate.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-stable-u"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detectors", nargs="+", default=list(STAGE1_DETECTORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--budgets", nargs="+", type=float, default=list(STAGE1_BUDGETS))
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(STAGE1_LAMBDAS))
    parser.add_argument("--main-lambda", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--min-gap", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--auto-budgets", action="store_true")
    parser.add_argument("--budget-multipliers", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--min-events-per-fold", type=float, default=5.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_stage1_gate(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detectors=tuple(args.detectors),
        seeds=tuple(args.seeds),
        budgets=tuple(args.budgets),
        lambdas=tuple(args.lambdas),
        main_lambda=args.main_lambda,
        k=args.k,
        gamma=args.gamma,
        fs=args.fs,
        min_gap=args.min_gap,
        n_folds=args.n_folds,
        calibration_fraction=args.calibration_fraction,
        auto_budgets=args.auto_budgets,
        budget_multipliers=tuple(args.budget_multipliers),
        min_events_per_fold=args.min_events_per_fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
