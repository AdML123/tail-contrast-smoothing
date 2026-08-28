from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.rrp import relaxation_features
from hbpc.score_benchmark import benchmark_score_vector
from hbpc.statistical_inference import rank_biserial
from hbpc.plotting import OKABE_ITO, apply_paper_style, save_paper_figure


@dataclass(frozen=True)
class SyntheticSetting:
    mu_normal: float
    mu_anomaly: float
    phi: float = 0.2
    noise_family: str = "gaussian"
    seed: int = 0
    length: int = 20_000
    events_per_class: int = 60
    tail_length: int = 12
    event_gap: int = 40
    peak_mean: float = 5.0
    peak_std: float = 0.10
    noise_scale: float = 0.18


@dataclass(frozen=True)
class SyntheticSeries:
    scores: np.ndarray
    labels: np.ndarray
    normal_event_starts: np.ndarray
    anomaly_event_starts: np.ndarray


def generate_synthetic_score_series(setting: SyntheticSetting) -> SyntheticSeries:
    rng = np.random.default_rng(setting.seed)
    scores = np.maximum(0.0, rng.normal(0.05, 0.02, size=setting.length)).astype(float)
    labels = np.zeros(setting.length, dtype=bool)
    starts = _sample_event_starts(
        rng,
        length=setting.length,
        count=setting.events_per_class * 2,
        min_gap=max(setting.event_gap, setting.tail_length + 4),
        margin=setting.tail_length + 2,
    )
    normal_starts = starts[::2][: setting.events_per_class]
    anomaly_starts = starts[1::2][: setting.events_per_class]
    peak_draws = rng.normal(setting.peak_mean, setting.peak_std, size=setting.events_per_class)
    for idx, start in enumerate(normal_starts):
        _write_event(scores, labels, start, peak_draws[idx], setting.mu_normal, setting, rng, anomalous=False)
    for idx, start in enumerate(anomaly_starts):
        _write_event(scores, labels, start, peak_draws[idx], setting.mu_anomaly, setting, rng, anomalous=True)
    return SyntheticSeries(
        scores=scores,
        labels=labels,
        normal_event_starts=np.asarray(normal_starts, dtype=int),
        anomaly_event_starts=np.asarray(anomaly_starts, dtype=int),
    )


def run_synthetic_regime_grid(
    output_dir: Path | str,
    mu_normals: Sequence[float] = (0.2, 0.8, 1.4, 2.0),
    mu_anomalies: Sequence[float] = (0.2, 0.8, 1.4, 2.0),
    phis: Sequence[float] = (0.0, 0.3, 0.6),
    noise_families: Sequence[str] = ("gaussian", "lognormal", "clipped"),
    seeds: Sequence[int] = (0, 1, 2),
    length: int = 20_000,
    events_per_class: int = 60,
    top_n: int = 180,
    k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for noise_family in noise_families:
        for phi in phis:
            for mu_normal in mu_normals:
                for mu_anomaly in mu_anomalies:
                    for seed in seeds:
                        setting = SyntheticSetting(
                            mu_normal=float(mu_normal),
                            mu_anomaly=float(mu_anomaly),
                            phi=float(phi),
                            noise_family=str(noise_family),
                            seed=int(seed),
                            length=int(length),
                            events_per_class=int(events_per_class),
                        )
                        series = generate_synthetic_score_series(setting)
                        metrics = benchmark_score_vector(
                            series.scores,
                            series.labels,
                            dataset="synthetic",
                            predictor="tail_location",
                            seed=int(seed),
                            top_ns=(int(top_n),),
                            windows=(int(k),),
                            ewma_alphas=(0.3,),
                        )
                        raw = metrics[metrics["postprocess"] == "raw"].iloc[0]
                        confirmation = metrics[metrics["postprocess"] == "confirmation_mean"].iloc[0]
                        features = relaxation_features(series.scores, k=int(k))
                        normal = features.relax[series.normal_event_starts]
                        anomaly = features.relax[series.anomaly_event_starts]
                        rows.append(
                            {
                                "noise_family": noise_family,
                                "phi": float(phi),
                                "mu_normal": float(mu_normal),
                                "mu_anomaly": float(mu_anomaly),
                                "delta_mu": float(mu_anomaly - mu_normal),
                                "seed": int(seed),
                                "replicate": int(seed),
                                "raw_f1_raw": float(raw["raw_f1"]),
                                "forward_f1_raw": float(confirmation["raw_f1"]),
                                "forward_gain": float(confirmation["raw_f1"] - raw["raw_f1"]),
                                "raw_pa_f1": float(raw["pa_f1"]),
                                "forward_pa_f1": float(confirmation["pa_f1"]),
                                "rank_biserial": rank_biserial(normal, anomaly),
                            }
                        )
    table = pd.DataFrame(rows)
    table.to_csv(tables_dir / "synthetic_regime_grid.csv", index=False)
    summary = _synthetic_summary(table)
    summary.to_csv(tables_dir / "synthetic_regime_summary.csv", index=False)
    heatmap = _synthetic_heatmap(summary)
    heatmap.to_csv(tables_dir / "synthetic_delta_gain_heatmap.csv", index=False)
    _plot_synthetic_heatmap(heatmap, figures_dir / "synthetic_delta_gain_heatmap.png")
    return table, heatmap


def _sample_event_starts(rng: np.random.Generator, length: int, count: int, min_gap: int, margin: int) -> np.ndarray:
    starts: list[int] = []
    attempts = 0
    while len(starts) < count and attempts < count * 500:
        attempts += 1
        candidate = int(rng.integers(margin, length - margin))
        if all(abs(candidate - existing) >= min_gap for existing in starts):
            starts.append(candidate)
    if len(starts) < count:
        raise RuntimeError("could not place synthetic events without overlap")
    return np.asarray(sorted(starts), dtype=int)


def _write_event(
    scores: np.ndarray,
    labels: np.ndarray,
    start: int,
    peak: float,
    tail_location: float,
    setting: SyntheticSetting,
    rng: np.random.Generator,
    anomalous: bool,
) -> None:
    scores[start] = max(scores[start], float(peak))
    if anomalous:
        labels[start : start + setting.tail_length + 1] = True
    previous = float(peak)
    for offset in range(1, setting.tail_length + 1):
        shock = _draw_noise(rng, setting.noise_family, setting.noise_scale)
        value = tail_location + setting.phi * (previous - tail_location) + shock
        scores[start + offset] = max(scores[start + offset], max(0.0, value))
        previous = value


def _draw_noise(rng: np.random.Generator, family: str, scale: float) -> float:
    if family == "gaussian":
        return float(rng.normal(0.0, scale))
    if family == "lognormal":
        return float(rng.lognormal(mean=-2.0, sigma=0.35) - np.exp(-2.0 + 0.35**2 / 2.0))
    if family == "clipped":
        return float(max(0.0, rng.normal(0.0, scale)))
    raise ValueError(f"unknown noise family: {family}")


def _synthetic_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["noise_family", "phi", "mu_normal", "mu_anomaly", "delta_mu"], as_index=False)
        .agg(
            raw_f1_raw=("raw_f1_raw", "mean"),
            forward_f1_raw=("forward_f1_raw", "mean"),
            forward_gain=("forward_gain", "mean"),
            gain_ci_low=("forward_gain", lambda x: float(np.quantile(x, 0.025))),
            gain_ci_high=("forward_gain", lambda x: float(np.quantile(x, 0.975))),
            n_replicates=("replicate", "nunique"),
            rank_biserial=("rank_biserial", "mean"),
            predicted_sign_fraction=("forward_gain", lambda x: float(np.mean(np.sign(x) == np.sign(x.mean())))),
        )
        .sort_values(["noise_family", "phi", "mu_normal", "mu_anomaly"])
    )


def _synthetic_heatmap(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["mu_normal", "mu_anomaly"], as_index=False)
        .agg(
            forward_gain=("forward_gain", "mean"),
            rank_biserial=("rank_biserial", "mean"),
            gain_ci_low=("gain_ci_low", "mean"),
            gain_ci_high=("gain_ci_high", "mean"),
            n_replicates=("n_replicates", "sum"),
            predicted_sign_fraction=("predicted_sign_fraction", "mean"),
        )
        .sort_values(["mu_normal", "mu_anomaly"])
    )
    return grouped


def _plot_synthetic_heatmap(heatmap: pd.DataFrame, output_path: Path) -> None:
    apply_paper_style()
    normals = sorted(heatmap["mu_normal"].unique())
    anomalies = sorted(heatmap["mu_anomaly"].unique())
    matrix = np.full((len(anomalies), len(normals)), np.nan, dtype=float)
    for _, row in heatmap.iterrows():
        y = anomalies.index(float(row["mu_anomaly"]))
        x = normals.index(float(row["mu_normal"]))
        matrix[y, x] = float(row["forward_gain"])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(matrix, origin="lower", cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(normals)), [f"{x:.1f}" for x in normals])
    ax.set_yticks(range(len(anomalies)), [f"{y:.1f}" for y in anomalies])
    ax.set_xlabel(r"normal-high tail location $\mu_N$")
    ax.set_ylabel(r"anomaly-high tail location $\mu_A$")
    ax.set_title("Forward-mean gain in controlled tails")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\Delta$ raw F1")
    save_paper_figure(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled synthetic short-tail regime experiments.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=20_000)
    parser.add_argument("--events-per-class", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=180)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    run_synthetic_regime_grid(
        output_dir=args.output_dir,
        length=int(args.length),
        events_per_class=int(args.events_per_class),
        top_n=int(args.top_n),
        k=int(args.k),
    )


if __name__ == "__main__":
    main()
