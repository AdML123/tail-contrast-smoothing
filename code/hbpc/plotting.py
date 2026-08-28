from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#666666",
}


def apply_paper_style() -> None:
    """Apply the shared plotting style used by the paper figures."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 220,
        }
    )


def legend_above(ax: Axes, ncol: int | None = None, *, y: float = 1.02) -> Legend | None:
    """Place an axes legend above the plotting area."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    columns = ncol if ncol is not None else min(max(len(handles), 1), 4)
    return ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=columns,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.8,
    )


def figure_legend_above(fig: Figure, axes: Iterable[Axes], ncol: int | None = None) -> Legend | None:
    """Place a de-duplicated figure legend above all subplots."""
    unique: dict[str, object] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    if not unique:
        return None
    columns = ncol if ncol is not None else min(max(len(unique), 1), 5)
    return fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=columns,
        frameon=False,
        handlelength=1.8,
    )


def save_paper_figure(fig: Figure, output_path: Path, *, top: float = 0.86) -> Path:
    """Save a figure with consistent margins for legends above the axes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_profiles(normal_profile: np.ndarray, anomaly_profile: np.ndarray, output_path: Path) -> Path:
    apply_paper_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    horizons = np.arange(1, len(normal_profile) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True, sharey=True)
    axes[0].plot(horizons, normal_profile, marker="o", color=OKABE_ITO["blue"], label="normal")
    axes[0].set_title("Normal")
    axes[0].set_ylabel("normalized error")
    axes[1].plot(horizons, anomaly_profile, marker="o", color=OKABE_ITO["vermillion"], label="anomaly")
    axes[1].set_title("Anomaly")
    axes[1].set_xlabel("horizon")
    axes[1].set_ylabel("normalized error")
    figure_legend_above(fig, axes, ncol=2)
    return save_paper_figure(fig, output_path)
