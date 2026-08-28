from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, plot_error_profiles


def test_paper_style_sets_accessible_palette_and_font_sizes() -> None:
    apply_paper_style()

    assert OKABE_ITO["blue"] == "#0072B2"
    assert plt.rcParams["font.size"] == 10
    assert plt.rcParams["axes.labelsize"] == 10
    assert plt.rcParams["legend.fontsize"] == 8


def test_legend_above_places_legend_outside_axes() -> None:
    apply_paper_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")

    legend = legend_above(ax)

    assert legend is not None
    assert legend.get_frame_on() is False
    assert legend._loc == 8  # lower center
    plt.close(fig)


def test_error_profiles_use_vertical_layout(tmp_path: Path) -> None:
    output = plot_error_profiles(
        np.array([1.0, 0.3, 0.2]),
        np.array([1.0, 0.8, 0.7]),
        tmp_path / "profiles.png",
    )

    assert output.is_file()


def test_legend_above_can_clear_axes_title() -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for idx in range(4):
        ax.plot([0, 1], [idx, idx + 1], label=f"series {idx}")
    title = ax.set_title("Title that must remain readable")

    legend = legend_above(ax, ncol=4, y=1.20)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    assert legend is not None
    assert title.get_window_extent(renderer).y1 < legend.get_window_extent(renderer).y0
    plt.close(fig)
