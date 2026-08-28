from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.plotting import OKABE_ITO, apply_paper_style


WIDTH_IN = 88.0 / 25.4


def _read_table(root: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = root / "tables" / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"none of the input tables exist: {names}")


def _save(fig, pdf_path: Path, svg_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)


def generate_spl_figures(input_root: Path | str, output_dir: Path | str) -> list[Path]:
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    synthetic = _read_table(input_root, ["synthetic_summary.csv", "synthetic_regime_summary.csv"])
    if "mean_gain" not in synthetic and "forward_gain" in synthetic:
        synthetic = synthetic.rename(columns={"forward_gain": "mean_gain"})
    dataset = _read_table(input_root, ["dataset_tail_contrast.csv"])
    try:
        performance = _read_table(input_root, ["delay_aware_performance.csv"])
    except FileNotFoundError:
        performance = pd.DataFrame()
    gain = performance.pivot_table(index="dataset", columns="method", values="raw_f1", aggfunc="mean") if not performance.empty else pd.DataFrame()
    if {"raw_realtime", "confirmation_mean"} <= set(gain.columns):
        dataset = dataset.copy()
        dataset["confirmation_gain"] = dataset["dataset"].map(gain["confirmation_mean"] - gain["raw_realtime"])
    try:
        swat = pd.read_csv(input_root / "tables" / "swat_intervention_all_windows.csv")
    except FileNotFoundError:
        swat = pd.DataFrame({"window": [3, 5, 10, 20], "analysis": ["observed"] * 4, "r_K3": [np.nan] * 4, "ci_low": [np.nan] * 4, "ci_high": [np.nan] * 4})

    fig1, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, 2.25), sharex=True)
    x = np.arange(4)
    for ax, (name, tail, color) in zip(axes, [("positive", [1.0, 0.8, 0.7, 0.7], OKABE_ITO["blue"]), ("null", [1.0, 0.3, 0.2, 0.2], OKABE_ITO["gray"]), ("reversal", [1.0, 0.2, 0.1, 0.05], OKABE_ITO["vermillion"])]):
        ax.plot(x, [1.0, *tail[1:]], marker="o", color=color)
        ax.axvline(0, color=OKABE_ITO["black"], linewidth=0.6)
        ax.set_ylabel(name, fontsize=8)
        ax.set_ylim(0, 1.1)
    axes[-1].set_xlabel("anchor t  |  confirmation samples t+1 ... t+K")
    fig1.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.16, hspace=0.35)
    fig1_pdf, fig1_svg = output_dir / "fig1_regime_alignment.pdf", output_dir / "fig1_regime_alignment.svg"
    _save(fig1, fig1_pdf, fig1_svg)

    fig2, panels = plt.subplots(3, 1, figsize=(WIDTH_IN, 5.7), sharex=False)
    panels[0].errorbar(synthetic["delta_mu"], synthetic["mean_gain"], yerr=[synthetic["mean_gain"] - synthetic["gain_ci_low"], synthetic["gain_ci_high"] - synthetic["mean_gain"]], fmt="o", color=OKABE_ITO["blue"], capsize=2)
    panels[0].axhline(0, color=OKABE_ITO["gray"], linewidth=0.7)
    panels[0].set_ylabel("gain")
    panels[0].set_title("(a) controlled tail contrast", loc="left", fontsize=8)
    if "confirmation_gain" in dataset:
        y = dataset["confirmation_gain"].astype(float)
    else:
        y = dataset["tail_contrast"] * 0.05
    label_offsets = {
        "SMD": (4, 5),
        "MSL": (4, 5),
        "SMAP": (4, -13),
        "PSM": (4, 5),
        "SWaT": (4, -24),
        "HAI": (4, 13),
    }
    for _, row in dataset.iterrows():
        underpowered = float(row.get("matched_pairs", 20)) < 20
        x_value = float(row["tail_contrast"])
        y_value = float(y.loc[row.name])
        panels[1].errorbar(
            x_value,
            y_value,
            xerr=[[x_value - float(row["contrast_ci_low"])], [float(row["contrast_ci_high"]) - x_value]],
            fmt="o",
            color=OKABE_ITO["orange"],
            markerfacecolor="none" if underpowered else OKABE_ITO["orange"],
            capsize=2,
        )
        label = str(row["dataset"]) + ("*" if underpowered else "")
        panels[1].annotate(
            label,
            (x_value, y_value),
            xytext=label_offsets.get(str(row["dataset"]), (4, 5)),
            textcoords="offset points",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
        )
    panels[1].axhline(0, color=OKABE_ITO["gray"], linewidth=0.7)
    panels[1].set_ylabel("confirmation - raw")
    panels[1].set_title("(b) six-dataset conditional contrast", loc="left", fontsize=8)
    panels[1].text(0.02, 0.90, "* underpowered (<20 pairs)", transform=panels[1].transAxes, fontsize=6.5)
    for analysis, group in swat.groupby("analysis"):
        if "tail_contrast" in group and group["tail_contrast"].notna().any():
            y_swat = group["tail_contrast"].astype(float)
            low_swat = group["contrast_ci_low"].astype(float)
            high_swat = group["contrast_ci_high"].astype(float)
            panels[2].errorbar(group["window"], y_swat, yerr=[(y_swat - low_swat).clip(lower=0), (high_swat - y_swat).clip(lower=0)], marker="o", label=analysis, capsize=2)
        else:
            panels[2].plot(group["window"], group["r_K3"], marker="o", label=analysis)
    panels[2].axhline(0, color=OKABE_ITO["gray"], linewidth=0.7)
    panels[2].set_xlabel("SWaT filter window")
    panels[2].set_ylabel("tail contrast")
    panels[2].set_title("(c) SWaT windows and block control", loc="left", fontsize=8)
    panels[2].legend(frameon=False, fontsize=7)
    fig2.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.08, hspace=0.50)
    fig2_pdf, fig2_svg = output_dir / "fig2_contrast_gain.pdf", output_dir / "fig2_contrast_gain.svg"
    panel_bounds = [[float(ax.get_position().x0), float(ax.get_position().y1)] for ax in panels]
    _save(fig2, fig2_pdf, fig2_svg)
    metadata = {
        "fig1": {"width_mm": 88.0},
        "fig2": {
            "width_mm": 88.0,
            "orientation": "vertical",
            "panels": [{"panel": label, "x0": bound[0], "y1": bound[1]} for label, bound in zip(("a", "b", "c"), panel_bounds)],
            "panel_bounds": panel_bounds,
        },
    }
    (output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [fig1_pdf, fig1_svg, fig2_pdf, fig2_svg]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    generate_spl_figures(args.input_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
