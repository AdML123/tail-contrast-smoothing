"""Draw a compact, single-column figure of the three gated tail cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.spl_experiments import load_run_with_training_scores
from hbpc.tail_contrast import extract_event_anchors, match_peak_anchors, normalize_scores


width_mm = 88.0
WIDTH_IN = width_mm / 25.4
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#222222"
GRAY = "#777777"
LIGHT_BLUE = "#B9D8EA"
LIGHT_ORANGE = "#F6D89B"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.2,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _matched_profiles(path: Path, k: int = 3) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    run, training = load_run_with_training_scores(path)
    normalized = normalize_scores(run.scores, training)
    normalized_training = normalize_scores(training[np.isfinite(training)], training[np.isfinite(training)])
    threshold = float(np.quantile(normalized_training, 0.99))
    anchors = extract_event_anchors(run.scores, run.labels, k, threshold, normalized)
    matched, balance = match_peak_anchors(anchors, caliper=0.2)
    if matched.empty:
        raise ValueError(f"no matched pairs in {path}")
    anomaly = []
    normal = []
    for row in matched.itertuples(index=False):
        anomaly.append(normalized[int(row.anomaly_index) : int(row.anomaly_index) + k + 1])
        normal.append(normalized[int(row.normal_index) : int(row.normal_index) + k + 1])
    return matched, np.asarray(anomaly), {"smd": float(balance["standardized_mean_difference"]), "contrast": float(matched["tail_difference"].mean())}


def _draw_dataset(ax, dataset: str, profiles: tuple[pd.DataFrame, np.ndarray, dict[str, float], np.ndarray], case: str) -> None:
    matched, anomaly, summary, _normal_profiles = profiles
    # The caller stores the normal profiles beside the anomaly profiles.
    normal_arr = np.asarray(profiles[3])
    x = np.arange(anomaly.shape[1])
    # Use one center per matched pair and one scale per dataset. Applying the
    # same affine transform to both classes preserves the tail-difference sign.
    pair_center = 0.5 * (normal_arr[:, [0]] + anomaly[:, [0]])
    normal_arr = normal_arr - pair_center
    anomaly = anomaly - pair_center
    pooled_tail = np.concatenate((normal_arr[:, 1:], anomaly[:, 1:]), axis=1)
    pair_scale = float(np.nanmedian(np.abs(pooled_tail)))
    normal_arr /= max(pair_scale, 1e-6)
    anomaly /= max(pair_scale, 1e-6)
    np.nan_to_num(normal_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(anomaly, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    selected = int(np.argmin(np.abs(matched["tail_difference"].to_numpy() - summary["contrast"])))
    for values, color, fill in ((normal_arr, ORANGE, LIGHT_ORANGE), (anomaly, BLUE, LIGHT_BLUE)):
        low, med, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        ax.fill_between(x, low, high, color=fill, alpha=0.55, linewidth=0)
        ax.plot(x, med, color=color, linewidth=0.7, alpha=0.65,
                linestyle="--" if color == ORANGE else "-")
        ax.plot(x, values[selected], color=color, linewidth=1.35, marker="o", markersize=2.2,
                markerfacecolor="white" if color == ORANGE else color,
                markeredgewidth=0.65, linestyle="--" if color == ORANGE else "-",
                label="normal tail" if color == ORANGE else "anomalous tail")
    ax.axvline(0, color=GRAY, linewidth=0.55)
    ax.axvline(3, color=GRAY, linewidth=0.55, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["peak", "t+1", "t+2", "t+3"], rotation=0)
    ax.set_title(f"{dataset} | {len(matched)} matched pairs", loc="left", pad=2)
    expected = {
        "positive": "predicted: anomalous higher",
        "null-compatible": "predicted: no resolved direction",
        "reversal": "predicted: normal higher",
    }[case]
    ax.text(0.98, 0.91, expected, transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=6.1)
    ax.set_ylim(-2.2, 1.0)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_ylabel("")


def generate_case_figure(score_root: Path, output_dir: Path, input_root: Path, k: int = 3) -> list[Path]:
    _style()
    contrasts = pd.read_csv(input_root / "tables" / "dataset_tail_contrast.csv").set_index("dataset")
    groups = {"positive": ["SMD", "PSM"], "null-compatible": ["HAI"], "reversal": ["MSL"]}
    loaded: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, float], np.ndarray]] = {}
    for dataset in sum(groups.values(), []):
        candidates = sorted((score_root / dataset / "one_step").glob("*/scores.npz"))
        if not candidates:
            raise FileNotFoundError(f"no score artifact for {dataset}")
        matched, anomaly, summary = _matched_profiles(candidates[0], k)
        run, training = load_run_with_training_scores(candidates[0])
        normalized = normalize_scores(run.scores, training)
        normal_profiles = np.asarray([normalized[int(row.normal_index) : int(row.normal_index) + k + 1] for row in matched.itertuples(index=False)])
        loaded[dataset] = (matched, anomaly, summary, normal_profiles)

    fig = plt.figure(figsize=(WIDTH_IN, 3.75))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.70, wspace=0.34,
                            left=0.17, right=0.98, top=0.88, bottom=0.12)
    axes = {}
    axes["SMD"] = fig.add_subplot(grid[0, 0])
    axes["PSM"] = fig.add_subplot(grid[0, 1], sharey=axes["SMD"])
    axes["HAI"] = fig.add_subplot(grid[1, :])
    axes["MSL"] = fig.add_subplot(grid[2, :])
    for dataset, case in (("SMD", "positive"), ("PSM", "positive"), ("HAI", "null-compatible"), ("MSL", "reversal")):
        _draw_dataset(axes[dataset], dataset, loaded[dataset], case)
    axes["PSM"].tick_params(labelleft=False)
    handles, labels = axes["SMD"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.965), ncol=2,
               frameon=False, handlelength=1.7, columnspacing=1.2)
    fig.text(0.025, 0.73, "positive", rotation=90, va="center", color=BLUE, fontsize=7.0)
    fig.text(0.025, 0.50, "null", rotation=90, va="center", color=GRAY, fontsize=7.0)
    fig.text(0.025, 0.24, "reversal", rotation=90, va="center", color="#D55E00", fontsize=7.0)
    fig.text(0.085, 0.49, "relative score", rotation=90, ha="center", va="center", fontsize=6.5, color=INK)
    fig.supxlabel("anchor and confirmation samples", x=0.58, y=0.015, fontsize=6.5, color=INK)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "fig2_case_instances.pdf"
    svg = output_dir / "fig2_case_instances.svg"
    png = output_dir / "fig2_case_instances.png"
    tiff = output_dir / "fig2_case_instances.tiff"  # release raster generated from the PNG preview
    fig.savefig(pdf)
    fig.savefig(svg)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    metadata = {
        "fig1": {"width_mm": width_mm},
        "fig2": {
        "width_mm": width_mm,
        "orientation": "vertical",
        "panels": ["positive:SMD", "positive:PSM", "null-compatible:HAI", "reversal:MSL"],
        "selection_rule": "closest matched pair to the dataset mean tail difference; bands show matched-pair interquartile ranges; both classes use a shared pair center and dataset scale",
        "underpowered_rows": ["SMAP", "SWaT"],
        },
    }
    (output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [pdf, svg, png]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_case_figure(args.score_root, args.output_dir, args.input_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
