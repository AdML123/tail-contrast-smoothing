"""Draw the three peak-matched tail cases at IEEE single-column width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from hbpc.spl_experiments import load_run_with_training_scores
from hbpc.tail_contrast import extract_event_anchors, match_peak_anchors, normalize_scores


WIDTH_MM = 88.0
WIDTH_IN = WIDTH_MM / 25.4
HEIGHT_IN = 4.35
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#202020"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D7D7D7"
BLUE_FILL = "#C7DFEE"
ORANGE_FILL = "#F5DCA8"
MIN_FONT_PT = 7.5
MIN_LINE_WIDTH_PT = 0.8
MIN_MARKER_SIZE_PT = 3.6
SELECTION_RULE = (
    "closest matched pair to the dataset mean tail difference; bands show "
    "matched-pair interquartile ranges; both classes use a shared pair center "
    "and dataset scale"
)

CASES = {
    "positive": ["SMD", "PSM"],
    "null_compatible": ["HAI"],
    "reversal": ["MSL"],
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": MIN_LINE_WIDTH_PT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _load_matched_profiles(path: Path, k: int) -> dict[str, object]:
    run, training = load_run_with_training_scores(path)
    finite_training = training[np.isfinite(training)]
    normalized = normalize_scores(run.scores, finite_training)
    normalized_training = normalize_scores(finite_training, finite_training)
    threshold = float(np.quantile(normalized_training, 0.99))
    anchors = extract_event_anchors(run.scores, run.labels, k, threshold, normalized)
    matched, balance = match_peak_anchors(anchors, caliper=0.2)
    if matched.empty:
        raise ValueError(f"no matched pairs in {path}")

    anomaly = np.asarray(
        [
            normalized[int(row.anomaly_index) : int(row.anomaly_index) + k + 1]
            for row in matched.itertuples(index=False)
        ],
        dtype=float,
    )
    normal = np.asarray(
        [
            normalized[int(row.normal_index) : int(row.normal_index) + k + 1]
            for row in matched.itertuples(index=False)
        ],
        dtype=float,
    )
    if anomaly.shape != normal.shape or anomaly.shape[1] != k + 1:
        raise ValueError(f"incomplete matched profile in {path}")

    pair_center = 0.5 * (normal[:, [0]] + anomaly[:, [0]])
    anomaly = anomaly - pair_center
    normal = normal - pair_center
    pooled_tail = np.concatenate((normal[:, 1:], anomaly[:, 1:]), axis=1)
    pair_scale = float(np.nanmedian(np.abs(pooled_tail)))
    if not np.isfinite(pair_scale) or pair_scale <= 1e-6:
        raise ValueError(f"degenerate shared profile scale in {path}")
    anomaly /= pair_scale
    normal /= pair_scale
    if not np.isfinite(anomaly).all() or not np.isfinite(normal).all():
        raise ValueError(f"non-finite matched profile in {path}")

    contrast = float(matched["tail_difference"].mean())
    selected = int(
        np.argmin(np.abs(matched["tail_difference"].to_numpy(dtype=float) - contrast))
    )
    return {
        "matched": matched,
        "anomaly": anomaly,
        "normal": normal,
        "selected": selected,
        "peak_smd": float(balance["standardized_mean_difference"]),
        "contrast": contrast,
        "pair_scale": pair_scale,
    }


def _draw_prediction(ax: plt.Axes, case: str) -> None:
    x = np.arange(4)
    profiles = {
        "positive": ([1.0, 0.78, 0.64, 0.54], [1.0, 0.38, 0.22, 0.14]),
        "null_compatible": ([1.0, 0.49, 0.34, 0.25], [1.0, 0.46, 0.32, 0.24]),
        "reversal": ([1.0, 0.28, 0.17, 0.11], [1.0, 0.72, 0.55, 0.43]),
    }
    anomaly, normal = profiles[case]
    ax.plot(
        x, anomaly, color=BLUE, linewidth=1.35, linestyle="-", marker="o",
        markersize=MIN_MARKER_SIZE_PT, markerfacecolor=BLUE, markeredgecolor=BLUE,
    )
    ax.plot(
        x, normal, color=ORANGE, linewidth=1.35, linestyle="--", marker="o",
        markersize=MIN_MARKER_SIZE_PT, markerfacecolor="white", markeredgecolor=ORANGE,
        markeredgewidth=MIN_LINE_WIDTH_PT,
    )
    ax.axvline(0, color=LIGHT_GRAY, linewidth=MIN_LINE_WIDTH_PT)
    ax.axvline(3, color=LIGHT_GRAY, linewidth=MIN_LINE_WIDTH_PT, linestyle=":")
    ax.set_xlim(-0.15, 3.90)
    ax.set_ylim(0.0, 1.12)
    ax.set_xticks([0, 1, 2, 3], ["p", "1", "2", "3"])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(3.12, anomaly[-1], "A", color=BLUE, va="center", ha="left", fontweight="bold")
    normal_y = normal[-1] - (0.055 if abs(anomaly[-1] - normal[-1]) < 0.08 else 0.0)
    ax.text(3.12, normal_y, "N", color=ORANGE, va="center", ha="left", fontweight="bold")


def _draw_link(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.annotate(
        "", xy=(0.92, 0.5), xytext=(0.08, 0.5), xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": MIN_LINE_WIDTH_PT},
    )


def _profile_limits(normal: np.ndarray, anomaly: np.ndarray, selected: int) -> tuple[float, float]:
    support = []
    for values in (normal, anomaly):
        low, _, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        support.extend((low, high, values[selected]))
    joined = np.concatenate(support)
    low = float(np.nanmin(joined))
    high = float(np.nanmax(joined))
    span = max(high - low, 0.8)
    return low - 0.10 * span, high + 0.14 * span


def _draw_empirical(
    ax: plt.Axes, dataset: str, profiles: dict[str, object], contrast_row: pd.Series
) -> None:
    anomaly = np.asarray(profiles["anomaly"], dtype=float)
    normal = np.asarray(profiles["normal"], dtype=float)
    selected = int(profiles["selected"])
    x = np.arange(anomaly.shape[1])

    for values, color, fill, linestyle, marker_face in (
        (anomaly, BLUE, BLUE_FILL, "-", BLUE),
        (normal, ORANGE, ORANGE_FILL, "--", "white"),
    ):
        low, median, high = np.nanpercentile(values, [25, 50, 75], axis=0)
        ax.fill_between(x, low, high, color=fill, alpha=0.48, linewidth=0)
        ax.plot(x, median, color=color, linewidth=MIN_LINE_WIDTH_PT, linestyle=linestyle, alpha=0.78)
        ax.plot(
            x, values[selected], color=color, linewidth=1.35, linestyle=linestyle,
            marker="o", markersize=MIN_MARKER_SIZE_PT, markerfacecolor=marker_face,
            markeredgecolor=color, markeredgewidth=MIN_LINE_WIDTH_PT,
        )

    ax.axvline(0, color=LIGHT_GRAY, linewidth=MIN_LINE_WIDTH_PT)
    ax.axvline(3, color=LIGHT_GRAY, linewidth=MIN_LINE_WIDTH_PT, linestyle=":")
    ax.set_xlim(-0.15, 3.90)
    ax.set_ylim(*_profile_limits(normal, anomaly, selected))
    ax.set_xticks([0, 1, 2, 3], ["p", "1", "2", "3"])
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis="both", length=2.0, pad=1.5)
    interval_note = "  CI crosses 0" if float(contrast_row["contrast_ci_low"]) <= 0 <= float(contrast_row["contrast_ci_high"]) else ""
    ax.set_title(
        f"{dataset}  m={len(profiles['matched'])}{interval_note}",
        loc="left",
        pad=2.0,
        fontweight="bold",
    )
    anomaly_end = float(anomaly[selected, -1])
    normal_end = float(normal[selected, -1])
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    if abs(anomaly_end - normal_end) < 0.08 * span:
        anomaly_end += 0.055 * span
        normal_end -= 0.055 * span
    ax.text(3.12, anomaly_end, "A", color=BLUE, va="center", ha="left", fontweight="bold")
    ax.text(3.12, normal_end, "N", color=ORANGE, va="center", ha="left", fontweight="bold")


def _artifact_for(score_root: Path, dataset: str) -> Path:
    candidates = sorted((score_root / dataset / "one_step").glob("*/scores.npz"))
    if not candidates:
        raise FileNotFoundError(f"no score artifact for {dataset}")
    return candidates[0]


def generate_case_figure(
    score_root: Path, output_dir: Path, input_root: Path, k: int = 3
) -> list[Path]:
    _style()
    contrast = pd.read_csv(input_root / "tables" / "dataset_tail_contrast.csv").set_index("dataset")
    required = {dataset for datasets in CASES.values() for dataset in datasets}
    missing = required - set(contrast.index.astype(str))
    if missing:
        raise ValueError(f"contrast table is missing datasets: {sorted(missing)}")

    loaded = {
        dataset: _load_matched_profiles(_artifact_for(score_root, dataset), k)
        for dataset in sorted(required)
    }

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    outer = fig.add_gridspec(
        3, 3, width_ratios=[0.88, 0.22, 1.55], height_ratios=[1.0, 1.0, 1.0],
        left=0.075, right=0.975, bottom=0.075, top=0.905, hspace=0.64, wspace=0.18,
    )
    case_labels = {
        "positive": "Positive",
        "null_compatible": "Null-compatible",
        "reversal": "Reversal",
    }
    prediction_axes: list[plt.Axes] = []
    empirical_axes_by_case: list[list[plt.Axes]] = []

    for row, case in enumerate(CASES):
        prediction_ax = fig.add_subplot(outer[row, 0])
        link_ax = fig.add_subplot(outer[row, 1])
        empirical_grid = outer[row, 2].subgridspec(1, len(CASES[case]), wspace=0.50)
        empirical_axes = [
            fig.add_subplot(empirical_grid[0, column])
            for column in range(len(CASES[case]))
        ]
        _draw_prediction(prediction_ax, case)
        _draw_link(link_ax)
        for ax, dataset in zip(empirical_axes, CASES[case]):
            _draw_empirical(ax, dataset, loaded[dataset], contrast.loc[dataset])

        prediction_axes.append(prediction_ax)
        empirical_axes_by_case.append(empirical_axes)
        y_center = 0.5 * (prediction_ax.get_position().y0 + prediction_ax.get_position().y1)
        fig.text(0.008, y_center, case_labels[case], rotation=90, ha="left", va="center", fontweight="bold")

    fig.text(0.245, 0.995, "Model prediction", ha="center", va="top", fontweight="bold")
    fig.text(0.705, 0.995, "Matched score tails", ha="center", va="top", fontweight="bold")
    fig.text(0.705, 0.955, "A = anomalous     N = normal", ha="center", va="top", color=INK)
    fig.supxlabel("peak anchor p and confirmation lag", x=0.64, y=0.012, fontsize=MIN_FONT_PT)

    for separator_row in (0, 1):
        upper_bottom = prediction_axes[separator_row].get_position().y0
        lower_top = prediction_axes[separator_row + 1].get_position().y1
        y = 0.5 * (upper_bottom + lower_top)
        fig.add_artist(
            plt.Line2D(
                [0.015, 0.985], [y, y], transform=fig.transFigure,
                color=LIGHT_GRAY, linewidth=0.6,
            )
        )

    fig.canvas.draw()
    case_bounds = []
    for prediction_ax, empirical_axes in zip(prediction_axes, empirical_axes_by_case):
        boxes = [prediction_ax.get_position(), *(ax.get_position() for ax in empirical_axes)]
        case_bounds.append(
            [
                float(min(box.x0 for box in boxes)),
                float(max(box.y1 for box in boxes)),
                float(max(box.x1 for box in boxes)),
                float(min(box.y0 for box in boxes)),
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "fig2_case_instances.pdf"
    svg = output_dir / "fig2_case_instances.svg"
    png = output_dir / "fig2_case_instances.png"
    tiff = output_dir / "fig2_case_instances.tiff"
    fig.savefig(svg, format="svg")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=600)
    fig.savefig(tiff, format="tiff", dpi=600)
    plt.close(fig)

    metadata = {
        "fig1": {"width_mm": WIDTH_MM},
        "fig2": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_IN * 25.4,
            "orientation": "vertical",
            "archetype": "schematic-led composite",
            "case_order": list(CASES),
            "case_bounds": case_bounds,
            "components": {case: ["prediction", "empirical"] for case in CASES},
            "datasets": CASES,
            "selection_rule": SELECTION_RULE,
            "context_statistic": "matched-pair median and interquartile range",
            "underpowered_rows": ["SMAP", "SWaT"],
            "minimum_font_pt": MIN_FONT_PT,
            "minimum_line_width_pt": MIN_LINE_WIDTH_PT,
            "minimum_marker_size_pt": MIN_MARKER_SIZE_PT,
            "encodings": {
                "anomalous": ["blue", "solid", "filled"],
                "normal": ["orange", "dashed", "open"],
            },
            "source_artifacts": {
                dataset: str(_artifact_for(score_root, dataset).relative_to(score_root))
                for dataset in sorted(required)
            },
        },
    }
    (output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return [pdf, svg, png, tiff]


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
