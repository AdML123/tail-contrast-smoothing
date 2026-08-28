from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.adaptation_causal import adaptation_features, compute_rank_biserial
from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure
from hbpc.score_benchmark import ScoreRun, benchmark_score_vector, load_npz_score_run
from hbpc.cluster_inference import paired_bootstrap_ci, unique_artifacts_by_sha256
from hbpc.tail_contrast import match_peak_anchors, normalize_scores, extract_event_anchors, paired_tail_contrast


def highpass_scores(scores: np.ndarray, window: int = 501) -> np.ndarray:
    if window <= 1:
        return np.asarray(scores, dtype=float).reshape(-1).copy()
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if window % 2 == 0:
        window += 1
    baseline = pd.Series(arr).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)
    return np.maximum(arr - baseline, 0.0)


def run_swat_filter_experiment(
    score_root: Path | str,
    output_dir: Path | str,
    dataset: str = "SWaT",
    method: str = "one_step",
    seeds: Sequence[int] = (0, 1, 2),
    windows: Sequence[int] = (101, 501, 1001),
    horizons: Sequence[int] = (3, 5),
    top_n: int = 300,
) -> pd.DataFrame:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        path = Path(score_root) / dataset / method / str(seed) / "scores.npz"
        if not path.exists():
            continue
        run = load_npz_score_run(path, dataset=dataset, predictor=method, seed=int(seed))
        variants: list[tuple[str, int, np.ndarray]] = [("none", 0, run.scores)]
        variants.extend(("highpass", int(window), highpass_scores(run.scores, window=int(window))) for window in windows)
        for filter_name, filter_window, scores in variants:
            adapt = adaptation_features(scores[:, None], run.labels, scores=scores, high_fraction=0.01)
            metrics = benchmark_score_vector(
                scores,
                run.labels,
                dataset=dataset,
                predictor=method,
                seed=int(seed),
                top_ns=(int(top_n),),
                windows=horizons,
                include_delayed_controls=True,
            )
            raw_best = metrics[metrics["postprocess"] == "raw"]["raw_f1"].max()
            forward_best = metrics[metrics["postprocess"] == "confirmation_mean"]["raw_f1"].max()
            row = {
                "dataset": dataset,
                "predictor": method,
                "seed": int(seed),
                "filter": filter_name,
                "filter_window": int(filter_window),
                "tau_anomaly_median": adapt["tau_anomaly_median"],
                "tau_normal_median": adapt["tau_normal_median"],
                "tau_ratio": adapt["tau_ratio"],
                "raw_best_f1": float(raw_best),
                "forward_best_f1": float(forward_best),
                "forward_gain": float(forward_best - raw_best),
            }
            for horizon in horizons:
                row[f"r_K{int(horizon)}"] = compute_rank_biserial(scores, run.labels, k=int(horizon), high_fraction=0.01)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "swat_highpass_summary.csv", index=False)
    _plot_regime_movement(table, figures_dir / "swat_highpass_regime_movement.png")
    return table


def run_swat_intervention(
    run,
    training_normal_scores: np.ndarray,
    output_dir: Path | str,
    windows: Sequence[int] = (3, 5, 10, 20),
    n_boot: int = 1_000,
    seed: int = 47,
) -> pd.DataFrame:
    """Evaluate all requested high-pass windows and a block-permuted control.

    The tail contrast is re-estimated after each transformation.  Confidence
    intervals use matched event-pair differences; the change relative to the
    block control is bootstrapped from the two event-pair samples.
    """
    output = Path(output_dir)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    normalized_reference = np.asarray(training_normal_scores, dtype=float)
    normalized_training = normalize_scores(normalized_reference, normalized_reference)

    def estimate(scores: np.ndarray) -> tuple[float, np.ndarray, int]:
        normalized = normalize_scores(scores, normalized_reference)
        threshold = float(np.quantile(normalized_training[np.isfinite(normalized_training)], 0.99))
        anchors = extract_event_anchors(scores, run.labels, 3, threshold, normalized)
        matched, balance = match_peak_anchors(anchors, caliper=0.2)
        if matched.empty:
            return float("nan"), np.empty(0, dtype=float), int(balance["matched_pairs"])
        differences = matched["tail_difference"].to_numpy(dtype=float)
        finite = differences[np.isfinite(differences)]
        return float(np.mean(finite)) if finite.size else float("nan"), finite, int(balance["matched_pairs"])

    def difference_interval(observed: np.ndarray, control: np.ndarray) -> tuple[float, float]:
        observed = observed[np.isfinite(observed)]
        control = control[np.isfinite(control)]
        if observed.size == 0 or control.size == 0:
            return float("nan"), float("nan")
        generator = np.random.default_rng(seed)
        observed_idx = generator.integers(0, observed.size, size=(n_boot, observed.size))
        control_idx = generator.integers(0, control.size, size=(n_boot, control.size))
        draws = observed[observed_idx].mean(axis=1) - control[control_idx].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        return float(low), float(high)

    rows: list[dict[str, object]] = []
    for window in windows:
        observed_scores = highpass_scores(run.scores, window=int(window))
        observed = float(compute_rank_biserial(observed_scores, run.labels, k=3, high_fraction=0.01))
        block_length = max(7, int(window))
        permuted = observed_scores.copy()
        blocks = [permuted[start : start + block_length].copy() for start in range(0, len(permuted), block_length)]
        rng.shuffle(blocks)
        control_scores = np.concatenate(blocks)[: len(permuted)]
        control = float(compute_rank_biserial(control_scores, run.labels, k=3, high_fraction=0.01))
        observed_contrast, observed_differences, observed_pairs = estimate(observed_scores)
        control_contrast, control_differences, control_pairs = estimate(control_scores)
        observed_ci = paired_bootstrap_ci(observed_differences, n_boot=n_boot, seed=seed) if observed_differences.size else {"ci_low": np.nan, "ci_high": np.nan}
        control_ci = paired_bootstrap_ci(control_differences, n_boot=n_boot, seed=seed) if control_differences.size else {"ci_low": np.nan, "ci_high": np.nan}
        delta = observed_contrast - control_contrast if np.isfinite(observed_contrast) and np.isfinite(control_contrast) else np.nan
        delta_low, delta_high = difference_interval(observed_differences, control_differences)
        for analysis, value, contrast, contrast_ci, matched_pairs in (("observed", observed, observed_contrast, observed_ci, observed_pairs), ("block_permuted_control", control, control_contrast, control_ci, control_pairs)):
            rows.append({
                "dataset": run.dataset, "predictor": run.predictor, "seed": run.seed,
                "window": int(window), "analysis": analysis, "r_K3": value,
                "delta_change": float(delta) if analysis == "observed" else 0.0,
                "delta_ci_low": float(delta_low), "delta_ci_high": float(delta_high),
                "tail_contrast": float(contrast), "contrast_ci_low": float(contrast_ci["ci_low"]), "contrast_ci_high": float(contrast_ci["ci_high"]),
                "ci_low": float(contrast_ci["ci_low"]), "ci_high": float(contrast_ci["ci_high"]),
                "matched_pairs": int(matched_pairs),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tables" / "swat_intervention_all_windows.csv", index=False)
    return frame


def run_swat_intervention_from_root(
    score_root: Path | str,
    output_dir: Path | str,
    seeds: Sequence[int] = (0, 1, 2),
    windows: Sequence[int] = (3, 5, 10, 20),
    n_boot: int = 1_000,
) -> pd.DataFrame:
    """Run the all-window intervention once per unique score artifact."""
    root, output = Path(score_root), Path(output_dir)
    paths = [root / "SWaT" / "one_step" / str(seed) / "scores.npz" for seed in seeds]
    paths = [path for path in paths if path.exists()]
    unique, _ = unique_artifacts_by_sha256(paths)
    frames: list[pd.DataFrame] = []
    for path in unique:
        with np.load(path) as payload:
            required = {"scores", "labels", "training_normal_scores"}
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            run = ScoreRun(
                dataset="SWaT", predictor="one_step", seed=int(path.parent.name),
                scores=np.asarray(payload["scores"], dtype=float), labels=np.asarray(payload["labels"]).astype(bool),
            )
            training = np.asarray(payload["training_normal_scores"], dtype=float)
        frame = run_swat_intervention(run, training, output, windows=windows, n_boot=n_boot, seed=47)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "tables" / "swat_intervention_all_windows.csv", index=False)
    return result


def _plot_regime_movement(table: pd.DataFrame, output_path: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    if not table.empty:
        summary = (
            table.groupby(["filter", "filter_window"], as_index=False)
            .agg(tau_normal_median=("tau_normal_median", "median"), r_K3=("r_K3", "mean"))
            .sort_values(["filter", "filter_window"])
        )
        colors = {"none": OKABE_ITO["gray"], "highpass": OKABE_ITO["blue"]}
        for filter_name, group in summary.groupby("filter"):
            ax.plot(
                group["tau_normal_median"],
                group["r_K3"],
                marker="o",
                color=colors.get(filter_name, OKABE_ITO["blue"]),
                label=filter_name,
            )
            for _, row in group.iterrows():
                label = "raw" if int(row["filter_window"]) == 0 else str(int(row["filter_window"]))
                ax.annotate(label, (float(row["tau_normal_median"]), float(row["r_K3"])), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"normal-high recovery time $\tau_N$")
    ax.set_ylabel(r"rank-biserial $r$ at $K=3$")
    legend_above(ax, ncol=2, y=1.08)
    save_paper_figure(fig, output_path, top=0.76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SWaT high-pass intervention for the short-tail location revision.")
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 10, 20])
    args = parser.parse_args()
    run_swat_filter_experiment(args.score_root, args.output_dir, seeds=args.seeds)
    run_swat_intervention_from_root(args.score_root, args.output_dir, seeds=args.seeds, windows=args.windows)


if __name__ == "__main__":
    main()
