from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.plotting import OKABE_ITO, apply_paper_style, save_paper_figure
from hbpc.rrp import assign_rrp_groups, relaxation_features
from hbpc.score_benchmark import load_npz_score_run
from hbpc.statistical_inference import (
    adjust_pvalues,
    bootstrap_ci,
    bootstrap_rank_biserial_ci,
    leave_one_out_spearman,
    permutation_test_rank_biserial,
)


def run_uncertainty_experiments(
    score_root: Path | str,
    output_dir: Path | str,
    adaptation_rows_path: Path | str | None = None,
    datasets: Sequence[str] = ("SMD", "MSL", "SMAP", "PSM", "SWaT"),
    methods: Sequence[str] = ("one_step",),
    seeds: Sequence[int] = (0, 1, 2),
    horizons: Sequence[int] = (3, 5, 10, 20),
    high_fraction: float = 0.01,
    n_boot: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rank = _rank_uncertainty(
        Path(score_root),
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        horizons=horizons,
        high_fraction=high_fraction,
        n_boot=n_boot,
    )
    if not rank.empty:
        rank["p_holm"] = adjust_pvalues(rank["p_perm"], method="holm")
        rank["p_bh"] = adjust_pvalues(rank["p_perm"], method="bh")
    rank.to_csv(tables_dir / "rank_biserial_uncertainty.csv", index=False)
    _plot_rank_forest(rank, figures_dir / "rank_biserial_ci_forest.png")

    tau = _tau_uncertainty(Path(adaptation_rows_path), n_boot=n_boot) if adaptation_rows_path else pd.DataFrame()
    tau.to_csv(tables_dir / "tau_uncertainty.csv", index=False)

    corr = _correlation_leave_one_out(Path(adaptation_rows_path)) if adaptation_rows_path else pd.DataFrame()
    corr.to_csv(tables_dir / "correlation_leave_one_out.csv", index=False)
    return rank, tau, corr


def _rank_uncertainty(
    score_root: Path,
    datasets: Sequence[str],
    methods: Sequence[str],
    seeds: Sequence[int],
    horizons: Sequence[int],
    high_fraction: float,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for method in methods:
            for horizon in horizons:
                normal_values: list[np.ndarray] = []
                anomaly_values: list[np.ndarray] = []
                for seed in seeds:
                    path = score_root / str(dataset) / str(method) / str(seed) / "scores.npz"
                    if not path.exists():
                        continue
                    run = load_npz_score_run(path, dataset=str(dataset), predictor=str(method), seed=int(seed))
                    if run.scores.size <= int(horizon):
                        continue
                    features = relaxation_features(run.scores, k=int(horizon))
                    groups = assign_rrp_groups(features, run.labels, k=int(horizon), high_fraction=high_fraction)
                    normal_values.append(features.relax[groups == "A"])
                    anomaly_values.append(features.relax[np.isin(groups, ("B1", "B2"))])
                if not normal_values or not anomaly_values:
                    continue
                normal = _finite_concat(normal_values)
                anomaly = _finite_concat(anomaly_values)
                ci = bootstrap_rank_biserial_ci(normal, anomaly, n_boot=n_boot, seed=13 + int(horizon))
                perm = permutation_test_rank_biserial(normal, anomaly, alternative="two-sided", n_permutations=min(20_000, n_boot * 20), seed=17)
                rows.append(
                    {
                        "dataset": dataset,
                        "predictor": method,
                        "k": int(horizon),
                        "n_A": int(normal.size),
                        "n_B": int(anomaly.size),
                        "rank_biserial": float(ci["estimate"]),
                        "ci_low": float(ci["ci_low"]),
                        "ci_high": float(ci["ci_high"]),
                        "p_perm": float(perm["p_value"]),
                        "permutation_mode": str(perm["mode"]),
                    }
                )
    return pd.DataFrame(rows)


def _tau_uncertainty(path: Path, n_boot: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if rows.empty:
        return pd.DataFrame()
    out: list[dict[str, object]] = []
    metrics = ["tau_anomaly_median", "tau_normal_median", "tau_ratio"]
    for (dataset, predictor), group in rows.groupby(["dataset", "predictor"]):
        for metric in metrics:
            if metric not in group:
                continue
            ci = bootstrap_ci(group[metric].to_numpy(dtype=float), statistic=np.median, n_boot=n_boot, seed=23)
            out.append(
                {
                    "dataset": dataset,
                    "predictor": predictor,
                    "metric": metric,
                    **ci,
                }
            )
    return pd.DataFrame(out)


def _correlation_leave_one_out(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if rows.empty:
        return pd.DataFrame()
    if "r_K3" not in rows:
        return pd.DataFrame()
    return leave_one_out_spearman(rows, feature="tau_normal_median", target="r_K3")


def _plot_rank_forest(rank: pd.DataFrame, output_path: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, max(2.8, 0.32 * max(len(rank), 1))))
    if not rank.empty:
        plot = rank.sort_values(["dataset", "k"]).reset_index(drop=True)
        y = np.arange(len(plot), dtype=float)
        estimates = plot["rank_biserial"].to_numpy(dtype=float)
        low = plot["ci_low"].to_numpy(dtype=float)
        high = plot["ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([estimates - low, high - estimates])
        ax.errorbar(
            estimates,
            y,
            xerr=xerr,
            fmt="o",
            color=OKABE_ITO["blue"],
            ecolor=OKABE_ITO["gray"],
            elinewidth=1.2,
            capsize=2.5,
        )
        labels = [f"{row.dataset}, K={int(row.k)}" for row in plot.itertuples()]
        ax.set_yticks(y, labels)
    ax.axvline(0.0, color=OKABE_ITO["vermillion"], linewidth=1.0, linestyle="--")
    ax.set_xlabel("rank-biserial separation with 95% bootstrap CI")
    ax.set_title("Uncertainty in short-tail separation")
    save_paper_figure(fig, output_path, top=0.93)


def _finite_concat(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    arr = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in values])
    return arr[np.isfinite(arr)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty analysis for the short-tail location revision.")
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adaptation-rows-path", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["SMD", "MSL", "SMAP", "PSM", "SWaT"])
    parser.add_argument("--methods", nargs="+", default=["one_step"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10, 20])
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    run_uncertainty_experiments(
        score_root=args.score_root,
        output_dir=args.output_dir,
        adaptation_rows_path=args.adaptation_rows_path,
        datasets=args.datasets,
        methods=args.methods,
        seeds=args.seeds,
        horizons=args.horizons,
        n_boot=int(args.n_boot),
    )


if __name__ == "__main__":
    main()
