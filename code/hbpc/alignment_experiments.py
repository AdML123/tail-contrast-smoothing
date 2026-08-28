from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbpc.score_benchmark import ScoreRun, benchmark_runs, load_npz_score_run
from hbpc.plotting import OKABE_ITO, apply_paper_style, legend_above, save_paper_figure


DEFAULT_METHODS = (
    "one_step",
    "multi_mean_raw",
    "multi_mean_norm_rms_clip",
    "hbpc_full_rms_clip",
)


def run_alignment_benchmark(
    scores_root: Path | str,
    output_dir: Path | str,
    dataset: str = "SMD",
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: Sequence[int] = (0, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores_root = Path(scores_root)
    output_dir = Path(output_dir)
    runs: list[ScoreRun] = []
    inventory_rows = _predictor_inventory()

    for method in methods:
        for seed in seeds:
            score_path = scores_root / dataset / method / str(seed) / "scores.npz"
            if score_path.exists():
                runs.append(load_npz_score_run(score_path, dataset=dataset, predictor=method, seed=int(seed)))
                inventory_rows.append(
                    {
                        "predictor": method,
                        "source": str(score_path),
                        "available": True,
                        "status": "local_score_file",
                    }
                )
            else:
                inventory_rows.append(
                    {
                        "predictor": method,
                        "source": str(score_path),
                        "available": False,
                        "status": "missing_score_file",
                    }
                )

    all_rows, best = benchmark_runs(runs, output_dir)
    inventory = pd.DataFrame(inventory_rows).drop_duplicates()
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(tables_dir / "predictor_inventory.csv", index=False)
    _plot_window_sensitivity(all_rows, figures_dir / "window_sensitivity_smd.png")
    return all_rows, best, inventory


def _predictor_inventory() -> list[dict[str, object]]:
    return [
        {
            "predictor": "OmniAnomaly",
            "source": "D:/HBPC/OmniAnomaly-master",
            "available": Path("D:/HBPC/OmniAnomaly-master").exists(),
            "status": "code_available_not_integrated",
        },
        {
            "predictor": "AnomalyTransformer",
            "source": "D:/HBPC/Anomaly-Transformer-main",
            "available": Path("D:/HBPC/Anomaly-Transformer-main").exists(),
            "status": "code_available_not_integrated",
        },
        {
            "predictor": "USAD",
            "source": "external_official_repo",
            "available": False,
            "status": "not_local",
        },
        {
            "predictor": "THOC",
            "source": "external_official_repo",
            "available": False,
            "status": "not_local",
        },
    ]


def _plot_window_sensitivity(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    subset = frame[(frame["postprocess"] == "forward_avg") & (frame["top_n"] == 300)]
    if subset.empty:
        return
    summary = (
        subset.groupby(["predictor", "k"], as_index=False)
        .agg(raw_f1=("raw_f1", "mean"), pa_f1=("pa_f1", "mean"), event_recall=("event_recall", "mean"))
        .sort_values(["predictor", "k"])
    )
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    palette = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], OKABE_ITO["purple"]]
    for idx, (predictor, group) in enumerate(summary.groupby("predictor")):
        ax.plot(group["k"], group["raw_f1"], marker="o", color=palette[idx % len(palette)], label=f"{predictor} raw F1")
    ax.set_xlabel("Forward window K")
    ax.set_ylabel("Raw F1 at top-N=300")
    ax.set_title("Window sensitivity on SMD")
    ax.grid(True, alpha=0.3)
    legend_above(ax, ncol=2)
    save_paper_figure(fig, output_path, top=0.82)

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run score-tail alignment benchmark on available local score files.")
    parser.add_argument("--scores-root", type=Path, default=Path("results-smd-gate") / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results-strr-alignment"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_alignment_benchmark(
        scores_root=args.scores_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
