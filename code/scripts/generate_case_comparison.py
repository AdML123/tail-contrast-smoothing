"""Generate a case comparison figure for score-tail post-processing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hbpc.plotting import OKABE_ITO, apply_paper_style, figure_legend_above, save_paper_figure
from hbpc.score_benchmark import forward_average_score, top_n_predictions


def _pick_normal_peak(scores: np.ndarray, labels: np.ndarray, k: int) -> int:
    candidates = np.flatnonzero((labels == 0) & np.isfinite(scores))
    if not candidates.size:
        raise ValueError("No normal candidates found")
    valid = candidates[candidates + k < len(scores)]
    if not valid.size:
        raise ValueError("No normal candidates with enough forward context")
    tail = forward_average_score(scores, k)
    ranked = sorted(valid, key=lambda idx: (scores[idx], -tail[idx]), reverse=True)
    return int(ranked[0])


def _pick_anomaly_peak(scores: np.ndarray, labels: np.ndarray, k: int) -> int:
    candidates = np.flatnonzero((labels > 0) & np.isfinite(scores))
    valid = candidates[candidates + k < len(scores)]
    if not valid.size:
        raise ValueError("No anomaly candidates with enough forward context")
    tail = forward_average_score(scores, k)
    ranked = sorted(valid, key=lambda idx: (tail[idx], scores[idx]), reverse=True)
    return int(ranked[0])


def _window(center: int, length: int, before: int, after: int) -> slice:
    start = max(0, int(center) - int(before))
    stop = min(length, int(center) + int(after) + 1)
    return slice(start, stop)


def _shade_labels(ax: plt.Axes, xs: np.ndarray, labels: np.ndarray) -> None:
    in_event = False
    start = 0
    for pos, value in enumerate(labels):
        if value and not in_event:
            start = pos
            in_event = True
        elif not value and in_event:
            ax.axvspan(xs[start] - 0.5, xs[pos - 1] + 0.5, color=OKABE_ITO["yellow"], alpha=0.24)
            in_event = False
    if in_event:
        ax.axvspan(xs[start] - 0.5, xs[len(labels) - 1] + 0.5, color=OKABE_ITO["yellow"], alpha=0.24)


def generate_case_comparison(
    scores_path: str | Path,
    output_dir: str | Path,
    k: int = 3,
    top_n: int = 300,
    radius_before: int = 12,
    radius_after: int = 18,
) -> Path:
    apply_paper_style()
    data = np.load(Path(scores_path))
    scores = np.asarray(data["scores"], dtype=float).reshape(-1)
    labels = np.asarray(data["labels"]).astype(int).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")

    forward = forward_average_score(scores, k)
    raw_alarm = top_n_predictions(scores, top_n=top_n, delay=0, length=len(scores))
    forward_alarm = top_n_predictions(forward, top_n=top_n, delay=k, length=len(scores))

    normal_center = _pick_normal_peak(scores, labels, k)
    anomaly_center = _pick_anomaly_peak(scores, labels, k)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharey=False)
    cases = [
        ("Normal high-score transient", normal_center),
        ("Anomalous persistent high-score event", anomaly_center),
    ]
    for ax, (title, center) in zip(axes, cases):
        sl = _window(center, len(scores), radius_before, radius_after)
        xs = np.arange(sl.start, sl.stop)
        local_scores = scores[sl]
        local_forward = forward[sl]
        local_labels = labels[sl]
        _shade_labels(ax, xs, local_labels)
        ax.plot(xs, local_scores, marker="o", color=OKABE_ITO["blue"], label="raw score")
        ax.plot(xs, local_forward, marker="s", color=OKABE_ITO["orange"], label=f"forward average K={k}")
        raw_hits = xs[raw_alarm[sl] > 0]
        fwd_hits = xs[forward_alarm[sl] > 0]
        if raw_hits.size:
            ax.scatter(raw_hits, scores[raw_hits], marker="x", s=60, color=OKABE_ITO["vermillion"], label="raw top-N alarm")
        if fwd_hits.size:
            ax.scatter(fwd_hits, scores[fwd_hits], marker="^", s=62, color=OKABE_ITO["green"], label="forward top-N alarm")
        ax.axvline(center, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0)
        ax.axvline(center + k, color=OKABE_ITO["green"], linestyle=":", linewidth=1.1)
        ax.set_title(title)
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time index")
    figure_legend_above(fig, axes, ncol=4)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case_comparison_smd_one_step.png"
    return save_paper_figure(fig, out_path, top=0.88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--radius-before", type=int, default=12)
    parser.add_argument("--radius-after", type=int, default=18)
    args = parser.parse_args()
    generate_case_comparison(
        scores_path=args.scores_path,
        output_dir=args.output_dir,
        k=args.k,
        top_n=args.top_n,
        radius_before=args.radius_before,
        radius_after=args.radius_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
