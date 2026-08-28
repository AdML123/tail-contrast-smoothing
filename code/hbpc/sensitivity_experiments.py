from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure
from hbpc.rrp import assign_rrp_groups, relaxation_features
from hbpc.score_benchmark import benchmark_score_vector, load_npz_score_run
from hbpc.supplement_experiments import _fast_rank_effect


def robust_normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr.copy()
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad if mad > eps else float(np.std(finite) + eps)
    return np.maximum((arr - median) / (scale + eps), 0.0)


def run_sensitivity_experiments(
    score_root: Path | str | None = None,
    output_dir: Path | str = "results-sensitivity",
    datasets: Sequence[str] = ("SMD", "MSL", "SMAP", "PSM", "SWaT"),
    methods: Sequence[str] = ("one_step",),
    seeds: Sequence[int] = (0, 1, 2),
    windows: Sequence[int] = (1, 2, 3, 5, 10, 20),
    top_ns: Sequence[int] = (100, 300, 500, 1000),
    high_fractions: Sequence[float] = (0.005, 0.01, 0.02, 0.05),
    score_roots: Mapping[str, Path | str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    metrics_dir = output / "metrics"
    figures_dir = output / "figures"
    for directory in (tables_dir, metrics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    roots = _normalise_score_roots(score_root=score_root, score_roots=score_roots)
    benchmark_rows: list[pd.DataFrame] = []
    phenomenon_rows: list[dict[str, object]] = []
    for split_scope, root in roots.items():
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    path = Path(root) / str(dataset) / str(method) / str(seed) / "scores.npz"
                    if not path.exists():
                        continue
                    run = load_npz_score_run(path, dataset=str(dataset), predictor=str(method), seed=int(seed))
                    variants = {
                        "raw": run.scores,
                        "robust": robust_normalize_scores(run.scores),
                    }
                    for normalization, scores in variants.items():
                        frame = benchmark_score_vector(
                            scores,
                            run.labels,
                            dataset=run.dataset,
                            predictor=run.predictor,
                            seed=run.seed,
                            top_ns=top_ns,
                            windows=windows,
                            include_delayed_controls=False,
                        )
                        frame["split_scope"] = split_scope
                        frame["normalization"] = normalization
                        benchmark_rows.append(frame)
                        feature_cache = {
                            int(k): relaxation_features(scores, k=int(k))
                            for k in windows
                            if len(scores) > int(k)
                        }
                        for k, features in feature_cache.items():
                            for high_fraction in high_fractions:
                                groups = assign_rrp_groups(features, run.labels, k=int(k), high_fraction=float(high_fraction))
                                effect = _fast_rank_effect(
                                    features.relax[groups == "A"],
                                    features.relax[np.isin(groups, ("B1", "B2"))],
                                )
                                phenomenon_rows.append(
                                    {
                                        "split_scope": split_scope,
                                        "dataset": dataset,
                                        "predictor": method,
                                        "seed": int(seed),
                                        "normalization": normalization,
                                        "high_fraction": float(high_fraction),
                                        "k": int(k),
                                        **effect,
                                    }
                                )
    benchmark = pd.concat(benchmark_rows, ignore_index=True) if benchmark_rows else pd.DataFrame()
    phenomenon = pd.DataFrame(phenomenon_rows)
    summary = _sensitivity_summary(benchmark)
    benchmark.to_csv(metrics_dir / "sensitivity_benchmark_rows.csv", index=False)
    phenomenon.to_csv(metrics_dir / "sensitivity_phenomenon_rows.csv", index=False)
    summary.to_csv(tables_dir / "sensitivity_summary.csv", index=False)
    _plot_sensitivity_curves(benchmark, figures_dir / "sensitivity_k_topn_curves.png")
    return benchmark, phenomenon, summary


def _normalise_score_roots(
    score_root: Path | str | None,
    score_roots: Mapping[str, Path | str] | None,
) -> dict[str, Path | str]:
    if score_roots:
        return {str(name): root for name, root in score_roots.items()}
    if score_root is None:
        raise ValueError("either score_root or score_roots must be provided")
    return {"capped": score_root}


def _sensitivity_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    if benchmark.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["split_scope", "dataset", "predictor", "normalization"] if "split_scope" in benchmark.columns else ["dataset", "predictor", "normalization"]
    for key, group in benchmark.groupby(group_columns):
        values = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        raw = group[group["postprocess"] == "raw"]
        forward = group[group["postprocess"] == "forward_avg"]
        ewma = group[group["postprocess"] == "ewma"]
        best_raw = float(raw["raw_f1"].max()) if not raw.empty else np.nan
        best_forward = float(forward["raw_f1"].max()) if not forward.empty else np.nan
        best_ewma = float(ewma["raw_f1"].max()) if not ewma.empty else np.nan
        rows.append(
            {
                **values,
                "best_raw_raw_f1": best_raw,
                "best_forward_raw_f1": best_forward,
                "best_ewma_raw_f1": best_ewma,
                "best_forward_gain": float(best_forward - best_raw) if np.isfinite(best_raw) and np.isfinite(best_forward) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_sensitivity_curves(benchmark: pd.DataFrame, output_path: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    if not benchmark.empty:
        keep = benchmark[(benchmark["postprocess"] == "forward_avg") & (benchmark["normalization"] == "raw")]
        if not keep.empty:
            grouped = (
                keep.groupby(["k", "top_n"], as_index=False)
                .agg(raw_f1=("raw_f1", "mean"))
                .sort_values(["top_n", "k"])
            )
            palette = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], OKABE_ITO["vermillion"]]
            for idx, (top_n, subset) in enumerate(grouped.groupby("top_n")):
                ax.plot(subset["k"], subset["raw_f1"], marker="o", color=palette[idx % len(palette)], label=f"top-N={int(top_n)}")
    ax.set_xscale("log")
    ax.set_xlabel("forward window K")
    ax.set_ylabel("raw F1")
    legend_above(ax, ncol=4, y=1.08)
    save_paper_figure(fig, output_path, top=0.76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short-tail location sensitivity controls.")
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--full-score-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["SMD", "MSL", "SMAP", "PSM", "SWaT"])
    parser.add_argument("--methods", nargs="+", default=["one_step"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    score_roots = {"capped": args.score_root}
    if args.full_score_root is not None:
        score_roots["full"] = args.full_score_root
    run_sensitivity_experiments(
        score_roots=score_roots,
        output_dir=args.output_dir,
        datasets=args.datasets,
        methods=args.methods,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
