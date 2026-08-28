from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hbpc.adaptation_causal import adaptation_features, compute_rank_biserial
from hbpc.data import load_dataset
from hbpc.experiments import predict_windows_streaming
from hbpc.linear import train_linear_predictor
from hbpc.plotting import OKABE_ITO, apply_paper_style, save_paper_figure
from hbpc.windows import make_training_windows, retrospective_errors


DATASETS = ("SMD", "MSL", "SMAP", "PSM", "SWaT")
HORIZONS = (3, 5)


def run_adaptation_correlation_from_arrays(
    runs: Sequence[tuple[str, str, int, np.ndarray, np.ndarray]],
    output_dir: Path | str,
    horizons: Sequence[int] = HORIZONS,
    high_fraction: float = 0.01,
    max_steps: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for dataset, predictor, seed, errors, labels in runs:
        scores = np.linalg.norm(np.nan_to_num(np.asarray(errors, dtype=np.float64), nan=0.0), axis=1)
        row = {
            "dataset": dataset,
            "predictor": predictor,
            "seed": int(seed),
            "source": "array_errors",
            **adaptation_features(
                errors,
                labels,
                scores=scores,
                max_steps=max_steps,
                high_fraction=high_fraction,
            ),
        }
        for horizon in horizons:
            row[f"r_K{int(horizon)}"] = compute_rank_biserial(
                scores,
                labels,
                k=int(horizon),
                high_fraction=high_fraction,
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    corr = _correlation_summary(frame, horizons=horizons)
    _write_outputs(Path(output_dir), frame, corr)
    return frame, corr


def run_adaptation_correlation_from_score_root(
    score_root: Path | str,
    output_dir: Path | str,
    datasets: Sequence[str] = DATASETS,
    methods: Sequence[str] = ("one_step",),
    seeds: Sequence[int] = (0, 1, 2),
    horizons: Sequence[int] = HORIZONS,
    high_fraction: float = 0.01,
    max_steps: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the diagnostic on existing score artifacts."""
    root = Path(score_root)
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                path = root / str(dataset) / str(method) / str(seed) / "scores.npz"
                if not path.exists():
                    continue
                with np.load(path) as payload:
                    scores = np.asarray(payload["scores"], dtype=np.float64).reshape(-1)
                    labels = np.asarray(payload["labels"]).astype(bool).reshape(-1)
                length = min(scores.size, labels.size)
                scores = scores[:length]
                labels = labels[:length]
                errors = scores[:, None]
                row = {
                    "dataset": dataset,
                    "predictor": method,
                    "seed": int(seed),
                    "source": "score_artifact",
                    **adaptation_features(
                        errors,
                        labels,
                        scores=scores,
                        max_steps=max_steps,
                        high_fraction=high_fraction,
                    ),
                }
                for horizon in horizons:
                    row[f"r_K{int(horizon)}"] = compute_rank_biserial(
                        scores,
                        labels,
                        k=int(horizon),
                        high_fraction=high_fraction,
                    )
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No score artifacts found under {root}")
    frame = pd.DataFrame(rows)
    corr = _correlation_summary(frame, horizons=horizons)
    _write_outputs(Path(output_dir), frame, corr)
    return frame, corr


def run_adaptation_correlation(
    data_root: Path | str,
    output_dir: Path | str,
    datasets: Sequence[str] = DATASETS,
    seeds: Sequence[int] = (0,),
    horizons: Sequence[int] = HORIZONS,
    lookback: int = 100,
    epochs: int = 10,
    learning_rate: float = 1e-2,
    train_size: int | None = 10000,
    test_size: int | None = 50000,
    device: str = "cpu",
    high_fraction: float = 0.01,
    max_steps: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs: list[tuple[str, str, int, np.ndarray, np.ndarray]] = []
    for dataset_name in datasets:
        dataset = load_dataset(Path(data_root), dataset_name)
        train = dataset.train[:train_size] if train_size is not None else dataset.train
        test = dataset.test[:test_size] if test_size is not None else dataset.test
        labels = dataset.labels[: len(test)]
        for seed in seeds:
            errors = _one_step_channel_errors(
                train=train,
                test=test,
                lookback=lookback,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=int(seed),
                device=device,
            )
            runs.append((dataset_name, "one_step_linear", int(seed), errors, labels))
    return run_adaptation_correlation_from_arrays(
        runs,
        output_dir=output_dir,
        horizons=horizons,
        high_fraction=high_fraction,
        max_steps=max_steps,
    )


def _one_step_channel_errors(
    train: np.ndarray,
    test: np.ndarray,
    lookback: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> np.ndarray:
    train_x, train_y = make_training_windows(train, lookback=lookback, horizons=1)
    model = train_linear_predictor(
        train_x,
        train_y,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        seed=int(seed),
        device=device,
    )
    predictions = predict_windows_streaming(model, test, lookback=lookback)
    errors = retrospective_errors(test, predictions, lookback=lookback, horizons=1)
    return np.asarray(errors[:, 0, :], dtype=np.float64)


def _correlation_summary(frame: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_names = [
        "tau_anomaly_median",
        "tau_normal_median",
        "tau_ratio",
        "peak_residual_ratio_median",
        "anomaly_segment_median_length",
        "normal_high_segment_median_length",
    ]
    for horizon in horizons:
        target = f"r_K{int(horizon)}"
        if target not in frame:
            continue
        for feature in feature_names:
            rows.append(_correlation_row(frame, feature=feature, target=target))
    return pd.DataFrame(rows)


def _correlation_row(frame: pd.DataFrame, feature: str, target: str) -> dict[str, object]:
    x = np.asarray(frame[feature], dtype=np.float64)
    y = np.asarray(frame[target], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    row = {"feature": feature, "target": target, "n": int(finite.sum())}
    if finite.sum() < 2 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        row.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan})
        return row
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson = pearsonr(x[finite], y[finite])
        spearman = spearmanr(x[finite], y[finite])
        row.update(
            {
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )
    except Exception:
        row.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan})
    return row


def _write_outputs(output_dir: Path, rows: pd.DataFrame, corr: pd.DataFrame) -> None:
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(metrics_dir / "adaptation_correlation_rows.csv", index=False)
    corr.to_csv(tables_dir / "adaptation_correlation_summary.csv", index=False)
    dataset_summary = _dataset_summary(rows)
    dataset_corr = _correlation_summary(dataset_summary, horizons=_available_horizons(rows))
    dataset_summary.to_csv(tables_dir / "adaptation_dataset_summary.csv", index=False)
    dataset_corr.to_csv(tables_dir / "adaptation_dataset_correlation_summary.csv", index=False)
    payload = {
        "rows": rows.to_dict(orient="records"),
        "correlation_analysis": corr.to_dict(orient="records"),
        "dataset_summary": dataset_summary.to_dict(orient="records"),
        "dataset_correlation_analysis": dataset_corr.to_dict(orient="records"),
    }
    (metrics_dir / "adaptation_correlation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    _plot_tau_vs_r(rows, figures_dir / "adaptation_tau_vs_r.png")


def _dataset_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    group_keys = [key for key in ["dataset", "predictor", "source"] if key in rows.columns]
    agg_map = {
        "tau_anomaly_median": "median",
        "tau_normal_median": "median",
        "tau_ratio": "median",
        "peak_residual_ratio_median": "median",
        "anomaly_segment_median_length": "median",
        "normal_high_segment_median_length": "median",
        "seed": "count",
    }
    for column in rows.columns:
        if column.startswith("r_K"):
            agg_map[column] = "mean"
    keep = {column: func for column, func in agg_map.items() if column in rows.columns}
    if not keep:
        return rows[group_keys].drop_duplicates().reset_index(drop=True)
    summary = rows.groupby(group_keys, as_index=False).agg(keep)
    if "seed" in summary.columns:
        summary = summary.rename(columns={"seed": "seeds"})
    return summary


def _available_horizons(rows: pd.DataFrame) -> tuple[int, ...]:
    horizons: list[int] = []
    for column in rows.columns:
        if column.startswith("r_K"):
            try:
                horizons.append(int(column[3:]))
            except ValueError:
                continue
    return tuple(sorted(horizons))


def _plot_tau_vs_r(rows: pd.DataFrame, output_path: Path) -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    if not rows.empty and {"tau_ratio", "r_K3"} <= set(rows.columns):
        plot_rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau_ratio", "r_K3"])
        if not plot_rows.empty:
            ax.scatter(
                plot_rows["tau_ratio"],
                plot_rows["r_K3"],
                s=36,
                color=OKABE_ITO["blue"],
                edgecolor="white",
                linewidth=0.6,
            )
            for _, row in plot_rows.iterrows():
                ax.annotate(
                    str(row["dataset"]),
                    (float(row["tau_ratio"]), float(row["r_K3"])),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.axhline(0.0, color=OKABE_ITO["gray"], linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"adaptation ratio $\tau_{\mathrm{anom}}/\tau_{\mathrm{normal}}$")
    ax.set_ylabel(r"relaxation separation $r$ at $K=3$")
    ax.set_title("Adaptation ratio vs. short-tail separation")
    save_paper_figure(fig, output_path, top=0.9)


def _json_default(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Adaptation speed vs short-tail location diagnostic.")
    parser.add_argument("--data-root", default=str(Path("datasets") / "Time-Series-Library"))
    parser.add_argument("--score-root", default="")
    parser.add_argument("--output-dir", default="results-adaptation-causal")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", default=["one_step"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    if args.score_root:
        rows, corr = run_adaptation_correlation_from_score_root(
            score_root=Path(args.score_root),
            output_dir=Path(args.output_dir),
            datasets=args.datasets,
            methods=args.methods,
            seeds=args.seeds,
            horizons=args.horizons,
        )
    else:
        rows, corr = run_adaptation_correlation(
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            datasets=args.datasets,
            seeds=args.seeds,
            horizons=args.horizons,
            lookback=args.lookback,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            train_size=args.train_size,
            test_size=args.test_size,
            device=args.device,
        )
    print(rows.to_string(index=False))
    print(corr.to_string(index=False))
    print(f"elapsed_seconds={time.perf_counter() - started:.2f}")


if __name__ == "__main__":
    main()
