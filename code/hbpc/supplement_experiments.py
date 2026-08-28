from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.rrp import assign_rrp_groups, relaxation_features
from hbpc.score_benchmark import ScoreRun, benchmark_score_vector, load_npz_score_run
from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure


DEFAULT_DATASETS = ("SMD", "MSL", "SMAP")
DEFAULT_METHODS = ("one_step",)
DEFAULT_TOP_NS = (100, 300, 500, 1000)
DEFAULT_WINDOWS = (1, 2, 3, 5, 10, 20)
DEFAULT_HORIZONS = (3, 5, 10, 20)


def run_supplement_experiments(
    score_roots: Sequence[Path | str],
    output_dir: Path | str,
    datasets: Sequence[str] = DEFAULT_DATASETS,
    benchmark_datasets: Sequence[str] | None = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: Sequence[int] = (0, 1, 2),
    top_ns: Sequence[int] = DEFAULT_TOP_NS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    phenomenon_horizons: Sequence[int] = DEFAULT_HORIZONS,
    high_fraction: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(score_roots, datasets=datasets, methods=methods, seeds=seeds)
    if not runs:
        raise FileNotFoundError("No score runs found for supplement experiments")

    benchmark_set = set(datasets if benchmark_datasets is None else benchmark_datasets)
    benchmark_runs = [run for run in runs if run.dataset in benchmark_set]
    if not benchmark_runs:
        raise FileNotFoundError("No score runs found for benchmark datasets")

    all_rows = pd.concat(
        [
            benchmark_score_vector(
                run.scores,
                run.labels,
                dataset=run.dataset,
                predictor=run.predictor,
                seed=run.seed,
                top_ns=top_ns,
                windows=windows,
                include_delayed_controls=True,
            )
            for run in benchmark_runs
        ],
        ignore_index=True,
    )
    all_rows.to_csv(metrics_dir / "supplement_all_rows.csv", index=False)

    budget_summary = _budget_curve_summary(all_rows)
    budget_summary.to_csv(tables_dir / "budget_curve_summary.csv", index=False)
    _plot_budget_curves(budget_summary, figures_dir)

    fairness = _delay_fairness_table(all_rows)
    fairness.to_csv(tables_dir / "delay_fairness.csv", index=False)

    phenomenon = _cross_dataset_phenomenon(
        runs,
        horizons=phenomenon_horizons,
        high_fraction=high_fraction,
    )
    phenomenon.to_csv(tables_dir / "cross_dataset_phenomenon.csv", index=False)
    _plot_cross_dataset_phenomenon(phenomenon, figures_dir / "cross_dataset_rank_biserial.png")

    best = _best_forward_vs_raw(all_rows)
    best.to_csv(tables_dir / "forward_vs_raw_best.csv", index=False)

    return all_rows, phenomenon, fairness


def _discover_runs(
    score_roots: Sequence[Path | str],
    datasets: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> list[ScoreRun]:
    runs: list[ScoreRun] = []
    seen: set[tuple[str, str, int]] = set()
    for root in [Path(p) for p in score_roots]:
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    key = (dataset, method, int(seed))
                    if key in seen:
                        continue
                    path = root / dataset / method / str(seed) / "scores.npz"
                    if path.exists():
                        runs.append(load_npz_score_run(path, dataset=dataset, predictor=method, seed=int(seed)))
                        seen.add(key)
    return runs


def _budget_curve_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame[frame["postprocess"].isin(["raw", "ewma", "backward_avg", "forward_avg"])]
    return (
        keep.groupby(["dataset", "predictor", "postprocess", "top_n", "k"], as_index=False)
        .agg(
            raw_f1=("raw_f1", "mean"),
            pa_f1=("pa_f1", "mean"),
            event_recall=("event_recall", "mean"),
            event_precision=("event_precision", "mean"),
            event_f1=("event_f1", "mean"),
            mttd=("mttd", "mean"),
            predicted_events=("predicted_events", "mean"),
        )
        .sort_values(["dataset", "predictor", "postprocess", "k", "top_n"])
    )


def _delay_fairness_table(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[frame["postprocess"].isin(["raw_delayed", "ewma_delayed", "backward_avg", "backward_avg_delayed", "forward_avg"])]
    subset = subset[subset["k"].isin([3, 5])]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby(["dataset", "predictor", "postprocess", "k", "top_n"], as_index=False)
        .agg(
            raw_f1=("raw_f1", "mean"),
            pa_f1=("pa_f1", "mean"),
            event_recall=("event_recall", "mean"),
            event_precision=("event_precision", "mean"),
            event_f1=("event_f1", "mean"),
            mttd=("mttd", "mean"),
            predicted_events=("predicted_events", "mean"),
        )
        .sort_values(["dataset", "predictor", "k", "top_n", "postprocess"])
    )


def _cross_dataset_phenomenon(
    runs: Sequence[ScoreRun],
    horizons: Sequence[int],
    high_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        for k in horizons:
            if run.scores.size <= int(k):
                continue
            features = relaxation_features(run.scores, k=int(k))
            groups = assign_rrp_groups(features, run.labels, k=int(k), high_fraction=high_fraction)
            first = features.relax[groups == "A"]
            second = features.relax[np.isin(groups, ("B1", "B2"))]
            effect = _fast_rank_effect(first, second)
            rows.append(
                {
                    "dataset": run.dataset,
                    "predictor": run.predictor,
                    "seed": run.seed,
                    "k": int(k),
                    **effect,
                }
            )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed
    aggregate_rows: list[dict[str, object]] = []
    for (dataset, predictor, k), group in per_seed.groupby(["dataset", "predictor", "k"]):
        aggregate_rows.append(
            {
                "dataset": dataset,
                "predictor": predictor,
                "seed": -1,
                "k": int(k),
                "n_A": int(group["n_A"].sum()),
                "n_B": int(group["n_B"].sum()),
                "median_A": float(group["median_A"].median()),
                "median_B": float(group["median_B"].median()),
                "median_ratio_B_over_A": float(group["median_ratio_B_over_A"].median()),
                "rank_biserial": float(group["rank_biserial"].mean()),
                "p_value": float(group["p_value"].max()),
            }
        )
    return pd.concat([per_seed, pd.DataFrame(aggregate_rows)], ignore_index=True)


def _best_forward_vs_raw(frame: pd.DataFrame) -> pd.DataFrame:
    keep = frame[frame["postprocess"].isin(["raw", "forward_avg", "ewma"])]
    rows: list[pd.Series] = []
    for (dataset, predictor, postprocess), group in keep.groupby(["dataset", "predictor", "postprocess"]):
        idx = group.sort_values(["raw_f1", "pa_f1", "event_recall"], ascending=[False, False, False]).index[0]
        row = group.loc[idx].copy()
        row["dataset"] = dataset
        row["predictor"] = predictor
        row["postprocess"] = postprocess
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()




def _fast_rank_effect(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first_arr = _finite_values(first)
    second_arr = _finite_values(second)
    n_first = int(first_arr.size)
    n_second = int(second_arr.size)
    if n_first == 0 or n_second == 0:
        return {
            "n_A": n_first,
            "n_B": n_second,
            "median_A": float("nan"),
            "median_B": float("nan"),
            "median_ratio_B_over_A": float("nan"),
            "rank_biserial": float("nan"),
            "p_value": float("nan"),
        }
    try:
        from scipy.stats import mannwhitneyu

        result = mannwhitneyu(second_arr, first_arr, alternative="two-sided")
        u = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        greater = float((second_arr[:, None] > first_arr[None, :]).sum())
        ties = float((second_arr[:, None] == first_arr[None, :]).sum())
        u = greater + 0.5 * ties
        p_value = float("nan")
    common = u / float(n_first * n_second)
    median_first = float(np.median(first_arr))
    median_second = float(np.median(second_arr))
    return {
        "n_A": n_first,
        "n_B": n_second,
        "median_A": median_first,
        "median_B": median_second,
        "median_ratio_B_over_A": float(median_second / median_first) if median_first != 0.0 else float("inf"),
        "rank_biserial": float(2.0 * common - 1.0),
        "p_value": p_value,
    }


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]

def _plot_budget_curves(summary: pd.DataFrame, figures_dir: Path) -> None:
    if summary.empty:
        return
    apply_paper_style()
    for (dataset, predictor), group in summary.groupby(["dataset", "predictor"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for postprocess, k, label, color, marker in [
            ("raw", 0, "raw", OKABE_ITO["gray"], "o"),
            ("ewma", 0, "EWMA", OKABE_ITO["green"], "s"),
            ("forward_avg", 3, "forward avg K=3", OKABE_ITO["blue"], "^"),
            ("forward_avg", 5, "forward avg K=5", OKABE_ITO["orange"], "D"),
        ]:
            subset = group[(group["postprocess"] == postprocess) & (group["k"] == k)]
            if subset.empty:
                continue
            ax.plot(subset["top_n"], subset["raw_f1"], marker=marker, color=color, label=label)
        ax.set_xlabel("Top-N alarm budget")
        ax.set_ylabel("Raw F1")
        ax.text(0.02, 0.94, f"{dataset} / {predictor}", transform=ax.transAxes, ha="left", va="top")
        ax.grid(True, alpha=0.3)
        legend_above(ax, ncol=4)
        save_paper_figure(fig, figures_dir / f"budget_curve_{dataset.lower()}_{predictor}.png", top=0.76)

def _plot_cross_dataset_phenomenon(phenomenon: pd.DataFrame, output_path: Path) -> None:
    subset = phenomenon[phenomenon["seed"] == -1]
    if subset.empty:
        return
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    palette = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        OKABE_ITO["vermillion"],
    ]
    for idx, (dataset, group) in enumerate(subset.groupby("dataset")):
        ordered = group.sort_values("k")
        ax.plot(ordered["k"], ordered["rank_biserial"], marker="o", color=palette[idx % len(palette)], label=dataset)
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    ax.set_xlabel("Relaxation horizon K")
    ax.set_ylabel("Rank-biserial r")
    ax.text(0.02, 0.94, "cross-dataset", transform=ax.transAxes, ha="left", va="top")
    ax.grid(True, alpha=0.3)
    legend_above(ax, ncol=5)
    save_paper_figure(fig, output_path, top=0.76)

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run supplemental short-tail location experiments.")
    parser.add_argument("--score-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results-strr-supplement"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--benchmark-datasets", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--top-ns", nargs="+", type=int, default=list(DEFAULT_TOP_NS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--phenomenon-horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--high-fraction", type=float, default=0.01)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_supplement_experiments(
        score_roots=tuple(args.score_roots),
        output_dir=args.output_dir,
        datasets=tuple(args.datasets),
        benchmark_datasets=None if args.benchmark_datasets is None else tuple(args.benchmark_datasets),
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        top_ns=tuple(args.top_ns),
        windows=tuple(args.windows),
        phenomenon_horizons=tuple(args.phenomenon_horizons),
        high_fraction=args.high_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
