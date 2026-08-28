from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure
from hbpc.rrp import (
    RankEffect,
    assign_rrp_groups,
    evaluate_phenomenon_gate,
    group_summary,
    rank_effect,
    relaxation_features,
)


HORIZONS = (3, 5, 10, 20)


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


def run_rrp_phenomenon_pilot(
    scores_root: Path | str = Path("results-smd-gate") / "raw",
    output_dir: Path | str = Path("results-rrp-smd-phenomenon"),
    dataset: str = "SMD",
    detector: str = "one_step",
    seeds: Sequence[int] = (0,),
    horizons: Sequence[int] = HORIZONS,
    high_fraction: float = 0.01,
    typical_quantiles: tuple[float, float] = (0.4, 0.6),
    min_group_size: int = 5,
    min_rank_biserial: float = 0.3,
    min_median_ratio: float = 1.5,
    min_pass_horizons: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    seed_effect_rows: list[dict[str, object]] = []
    aggregate_by_k: dict[int, dict[str, list[np.ndarray]]] = {
        int(k): {"A": [], "B": [], "B2": []} for k in horizons
    }
    curve_by_k: dict[int, dict[str, list[np.ndarray]]] = {
        int(k): {"A": [], "B": [], "B2": []} for k in horizons
    }

    for seed in seeds:
        run = load_score_run(scores_root / dataset / detector / str(seed))
        for horizon in horizons:
            k = int(horizon)
            features = relaxation_features(run.scores, k=k)
            groups = assign_rrp_groups(
                features,
                run.labels,
                k=k,
                high_fraction=high_fraction,
                typical_quantiles=typical_quantiles,
            )
            summary_frames.append(
                group_summary(
                    groups,
                    features.relax,
                    features.peak,
                    features.tail,
                    k=k,
                    seed=int(seed),
                )
            )
            for comparison, first_group, second_groups in (
                ("A_vs_B", "A", ("B1", "B2")),
                ("A_vs_B2", "A", ("B2",)),
            ):
                first = features.relax[groups == first_group]
                second = features.relax[np.isin(groups, second_groups)]
                effect = rank_effect(first, second)
                seed_effect_rows.append(_effect_row(dataset, detector, int(seed), k, comparison, effect))

            for group_name, members in {
                "A": ("A",),
                "B": ("B1", "B2"),
                "B2": ("B2",),
            }.items():
                mask = np.isin(groups, members)
                aggregate_by_k[k][group_name].append(features.relax[mask])
                curve_by_k[k][group_name].append(features.curves[mask])

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    aggregate_effect_rows, effects_by_k = _aggregate_effect_rows(dataset, detector, aggregate_by_k)
    effects = pd.DataFrame([*seed_effect_rows, *aggregate_effect_rows])
    gate = evaluate_phenomenon_gate(
        effects_by_k,
        min_group_size=min_group_size,
        min_rank_biserial=min_rank_biserial,
        min_median_ratio=min_median_ratio,
        min_pass_horizons=min_pass_horizons,
    )

    summary.to_csv(tables_dir / "rrp_group_summary.csv", index=False)
    effects.to_csv(tables_dir / "rrp_effects.csv", index=False)
    gate.to_csv(tables_dir / "rrp_gate.csv", index=False)
    for horizon in horizons:
        k = int(horizon)
        _plot_ecdf(aggregate_by_k[k], figures_dir / f"relax_ecdf_K{k}.png", k)
        _plot_relaxation_curve(curve_by_k[k], figures_dir / f"relaxation_curve_K{k}.png", k)
    return summary, effects, gate


def _aggregate_effect_rows(
    dataset: str,
    detector: str,
    values_by_k: dict[int, dict[str, list[np.ndarray]]],
) -> tuple[list[dict[str, object]], dict[int, dict[str, RankEffect]]]:
    rows: list[dict[str, object]] = []
    effects_by_k: dict[int, dict[str, RankEffect]] = {}
    for k, groups in values_by_k.items():
        a_values = _concat(groups["A"])
        b_values = _concat(groups["B"])
        b2_values = _concat(groups["B2"])
        effects_by_k[k] = {
            "A_vs_B": rank_effect(a_values, b_values),
            "A_vs_B2": rank_effect(a_values, b2_values),
        }
        for comparison, effect in effects_by_k[k].items():
            rows.append(_effect_row(dataset, detector, seed=-1, k=k, comparison=comparison, effect=effect))
    return rows, effects_by_k


def _effect_row(
    dataset: str,
    detector: str,
    seed: int,
    k: int,
    comparison: str,
    effect: RankEffect,
) -> dict[str, object]:
    row = asdict(effect)
    row.update(
        {
            "dataset": dataset,
            "detector": detector,
            "seed": seed,
            "k": k,
            "comparison": comparison,
        }
    )
    return row


def _concat(chunks: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(chunk, dtype=float).reshape(-1) for chunk in chunks if np.asarray(chunk).size]
    return np.concatenate(valid) if valid else np.array([], dtype=float)


def _plot_ecdf(values_by_group: dict[str, list[np.ndarray]], output_path: Path, k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    styles = [
        ("A", "normal-high", OKABE_ITO["blue"]),
        ("B", "anomaly-high", OKABE_ITO["orange"]),
        ("B2", "persistent anomaly-high", OKABE_ITO["green"]),
    ]
    for group, label, color in styles:
        values = np.sort(_concat(values_by_group[group]))
        if values.size == 0:
            continue
        y = np.arange(1, values.size + 1) / values.size
        ax.step(values, y, where="post", color=color, label=label)
    ax.text(0.98, 0.06, f"K={k}", transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xlabel("relaxation ratio")
    ax.set_ylabel("ECDF")
    legend_above(ax, ncol=3, y=1.08)
    save_paper_figure(fig, output_path, top=0.76)

def _plot_relaxation_curve(curves_by_group: dict[str, list[np.ndarray]], output_path: Path, k: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    offsets = np.arange(k + 1)
    for group, label, color in (
        ("A", "normal-high", OKABE_ITO["blue"]),
        ("B", "anomaly-high", OKABE_ITO["orange"]),
    ):
        chunks = [np.asarray(chunk, dtype=float) for chunk in curves_by_group[group] if np.asarray(chunk).size]
        if not chunks:
            continue
        curves = np.vstack(chunks)
        ax.plot(offsets, np.nanmean(curves, axis=0), marker="o", color=color, label=label)
    ax.text(0.98, 0.06, f"K={k}", transform=ax.transAxes, ha="right", va="bottom")
    ax.set_xlabel("offset")
    ax.set_ylabel("mean normalized score")
    legend_above(ax, ncol=2, y=1.08)
    save_paper_figure(fig, output_path, top=0.76)

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RRP SMD phenomenon pilot.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-rrp-smd-phenomenon"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--detector", default="one_step")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--high-fraction", type=float, default=0.01)
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--min-rank-biserial", type=float, default=0.3)
    parser.add_argument("--min-median-ratio", type=float, default=1.5)
    parser.add_argument("--min-pass-horizons", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_rrp_phenomenon_pilot(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        detector=args.detector,
        seeds=tuple(args.seeds),
        horizons=tuple(args.horizons),
        high_fraction=args.high_fraction,
        min_group_size=args.min_group_size,
        min_rank_biserial=args.min_rank_biserial,
        min_median_ratio=args.min_median_ratio,
        min_pass_horizons=args.min_pass_horizons,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
