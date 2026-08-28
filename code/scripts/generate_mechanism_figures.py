"""Generate mechanism figures for the short-tail location paper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure


RELAXATION_LEGEND_LABELS = ("normal transient", "persistent anomaly", "short tail K=3")
FORWARD_AVERAGING_LEGEND_LABELS = ("future tail", "candidate at t", "alarm at t+K")


def generate_mechanism_figures(output_dir: str | Path) -> None:
    apply_paper_style()
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(-2, 7)
    normal = np.array([1, 1, 9, 1.4, 1.1, 1.0, 1.0, 1.0, 1.0])
    anomaly = np.array([1, 1, 9, 7.2, 6.0, 4.8, 3.8, 3.2, 2.8])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, normal, marker="o", color=OKABE_ITO["blue"], label=RELAXATION_LEGEND_LABELS[0])
    ax.plot(x, anomaly, marker="o", color=OKABE_ITO["vermillion"], label=RELAXATION_LEGEND_LABELS[1])
    ax.axvline(0, color=OKABE_ITO["black"], linestyle="--", linewidth=1.1)
    ax.text(0.08, 8.6, "same peak", fontsize=10, va="top")
    ax.axvspan(1, 3, color=OKABE_ITO["sky"], alpha=0.25, label=RELAXATION_LEGEND_LABELS[2])
    ax.annotate("fast relaxation", xy=(1, normal[3]), xytext=(2.2, 3.0), arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
    ax.annotate("slow relaxation", xy=(2, anomaly[4]), xytext=(3.5, 7.4), arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10)
    ax.set_xlabel("offset from score peak")
    ax.set_ylabel("anomaly score")
    ax.set_xticks(x)
    ax.set_ylim(0, 10)
    legend_above(ax, ncol=3)
    save_paper_figure(fig, fig_dir / "mechanism_relaxation_toy.png", top=0.80)

    scores = np.array([0.8, 1.0, 8.8, 7.0, 5.8, 4.7, 2.2, 1.3])
    t = np.arange(scores.size)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(t, scores, marker="o", color=OKABE_ITO["blue"])
    ax.bar(t[3:6], scores[3:6], width=0.55, color=OKABE_ITO["orange"], alpha=0.55, label=FORWARD_AVERAGING_LEGEND_LABELS[0])
    ax.scatter([2], [scores[2]], s=90, color=OKABE_ITO["vermillion"], zorder=5, label=FORWARD_AVERAGING_LEGEND_LABELS[1])
    ax.scatter([5], [scores[5]], s=90, color=OKABE_ITO["green"], zorder=5, label=FORWARD_AVERAGING_LEGEND_LABELS[2])
    ax.add_patch(Rectangle((2.65, 0), 2.7, 8.0, fill=False, edgecolor=OKABE_ITO["orange"], linewidth=2.0, linestyle="--"))
    ax.annotate(r"$\bar{s}_t(K)=K^{-1}\sum_{k=1}^{K}s_{t+k}$", xy=(4, 6.0), xytext=(3.1, 8.5), arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=11)
    arrow = FancyArrowPatch((2, 0.8), (5, 0.8), arrowstyle="<->", mutation_scale=14, linewidth=1.4, color=OKABE_ITO["black"])
    ax.add_patch(arrow)
    ax.text(3.5, 1.05, "confirmation delay K", ha="center", fontsize=10)
    ax.set_xlabel("time index")
    ax.set_ylabel("anomaly score")
    ax.set_ylim(0, 10)
    ax.set_xticks(t)
    legend_above(ax, ncol=3)
    save_paper_figure(fig, fig_dir / "mechanism_forward_average.png", top=0.80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path("paper-strr-elsarticle") / "figures"),
        help="Directory where mechanism PNG files are written.",
    )
    args = parser.parse_args()
    generate_mechanism_figures(args.output_dir)
