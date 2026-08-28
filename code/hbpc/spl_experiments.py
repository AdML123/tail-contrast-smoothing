from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from hbpc.cluster_inference import event_cluster_bootstrap_metrics, paired_bootstrap_ci, unique_artifacts_by_sha256
from hbpc.score_benchmark import ScoreRun, confirmation_window_score, ewma_score, top_n_predictions
from hbpc.tail_contrast import extract_event_anchors, match_peak_anchors, normalize_scores, paired_tail_contrast


@dataclass(frozen=True)
class PrimaryProtocol:
    k: int = 3
    alarm_fraction: float = 0.005
    ewma_alpha: float = 0.3
    normal_peak_quantile: float = 0.99
    peak_caliper: float = 0.2
    practical_null: float = 0.1
    n_boot: int = 10_000
    seed: int = 47


PRIMARY_PROTOCOL = PrimaryProtocol()


def load_run_with_training_scores(path: Path) -> tuple[ScoreRun, np.ndarray]:
    with np.load(path) as payload:
        required = {"scores", "labels", "training_normal_scores"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        parts = Path(path).parts
        scores = np.asarray(payload["scores"], dtype=float).reshape(-1)
        labels = np.asarray(payload["labels"]).astype(bool).reshape(-1)
        segment_lengths = tuple(np.asarray(payload["segment_lengths"], dtype=int).reshape(-1)) if "segment_lengths" in payload else None
        if segment_lengths is not None and (any(length <= 0 for length in segment_lengths) or sum(segment_lengths) != len(scores)):
            raise ValueError(f"{path} has invalid segment_lengths")
        if len(labels) != len(scores):
            raise ValueError(f"{path} scores and labels have different lengths")
        run = ScoreRun(
            dataset=parts[-4], predictor=parts[-3], seed=int(parts[-2]) if parts[-2].isdigit() else 0,
            scores=scores,
            labels=labels,
            segment_lengths=segment_lengths,
        )
        training = np.asarray(payload["training_normal_scores"], dtype=float).reshape(-1)
    return run, training


def load_unique_score_runs(paths: Sequence[Path]) -> tuple[list[tuple[ScoreRun, np.ndarray]], dict[str, object]]:
    unique, duplicates = unique_artifacts_by_sha256(paths)
    runs = [load_run_with_training_scores(path) for path in unique]
    return runs, {"input_artifacts": len(paths), "unique_artifacts": len(unique), "duplicates": {str(path): str(source) for path, source in duplicates.items()}}


def fixed_alarm_count(length: int, fraction: float) -> int:
    if length <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("length and fraction must define a positive budget")
    return max(1, int(np.ceil(length * fraction)))


def selected_alarm_indices(scores: np.ndarray, fraction: float) -> np.ndarray:
    return np.flatnonzero(top_n_predictions(scores, top_n=fixed_alarm_count(len(scores), fraction)))


def classify_regime(ci_low: float, ci_high: float, null: float = 0.1) -> str:
    if ci_low > null:
        return "positive"
    if ci_high < -null:
        return "reversal"
    return "null_or_uncertain"


def classify_evidence(matched_pairs: int, minimum_pairs: int = 20) -> str:
    if matched_pairs >= minimum_pairs:
        return "primary"
    return "exploratory_underpowered"


def _segmentwise_transform(
    scores: np.ndarray,
    segment_lengths: tuple[int, ...] | None,
    transform,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if segment_lengths is None:
        return np.asarray(transform(arr), dtype=float)
    lengths = tuple(int(length) for length in segment_lengths)
    if any(length <= 0 for length in lengths) or sum(lengths) != arr.size:
        raise ValueError("segment_lengths must be positive and sum to the score length")
    offsets = np.cumsum((0, *lengths))
    pieces = [np.asarray(transform(arr[offsets[index] : offsets[index + 1]]), dtype=float) for index in range(len(lengths))]
    return np.concatenate(pieces)


def analyze_dataset(run: ScoreRun, training_normal_scores: np.ndarray, protocol: PrimaryProtocol = PRIMARY_PROTOCOL) -> pd.DataFrame:
    normalized = normalize_scores(run.scores, training_normal_scores)
    finite_training = np.asarray(training_normal_scores, dtype=float)
    finite_training = finite_training[np.isfinite(finite_training)]
    normalized_training = normalize_scores(finite_training, finite_training)
    threshold = float(np.quantile(normalized_training, protocol.normal_peak_quantile))
    anchors = extract_event_anchors(run.scores, run.labels, protocol.k, threshold, normalized)
    matched, balance = match_peak_anchors(anchors, protocol.peak_caliper)
    if matched.empty:
        raise ValueError("no peak-matched event pairs were retained")
    contrast = paired_tail_contrast(matched)
    interval = paired_bootstrap_ci(matched["tail_difference"].to_numpy(), protocol.n_boot, protocol.seed)
    alarm_count = fixed_alarm_count(run.scores.size, protocol.alarm_fraction)
    methods = {
        "raw_realtime": (run.scores, 0),
        "raw_delayed": (run.scores, protocol.k),
        "confirmation_mean": (_segmentwise_transform(run.scores, run.segment_lengths, lambda values: confirmation_window_score(values, protocol.k)), 0),
        "ewma": (_segmentwise_transform(run.scores, run.segment_lengths, lambda values: ewma_score(values, protocol.ewma_alpha)), 0),
    }
    rows: list[dict[str, object]] = []
    for method, (score, delay) in methods.items():
        predictions = top_n_predictions(score, alarm_count, delay=delay, length=run.labels.size)
        evidence_status = classify_evidence(int(balance["matched_pairs"]))
        row = {
            "dataset": run.dataset, "predictor": run.predictor, "seed": run.seed, "method": method,
            "matched_pairs": int(balance["matched_pairs"]), "peak_smd": float(balance["standardized_mean_difference"]),
            "tail_contrast": float(contrast["tail_contrast"]), "contrast_ci_low": float(interval["ci_low"]), "contrast_ci_high": float(interval["ci_high"]),
            "regime": classify_regime(float(interval["ci_low"]), float(interval["ci_high"]), protocol.practical_null) if evidence_status == "primary" else "underpowered",
            "evidence_status": evidence_status,
        }
        metrics = _metric_point(run.labels, predictions)
        uncertainty = event_cluster_bootstrap_metrics(run.labels, predictions, protocol.n_boot, protocol.seed)
        for name, value in metrics.items():
            row[name] = value
            row[f"{name}_ci_low"] = uncertainty["metrics"][name]["ci_low"]
            row[f"{name}_ci_high"] = uncertainty["metrics"][name]["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_point(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from hbpc.metrics import detection_delay, events_from_binary, f1

    events = events_from_binary(np.asarray(labels).astype(bool))
    recall = float(np.mean([predictions[start : stop + 1].any() for start, stop in events])) if events else 0.0
    return {"raw_f1": float(f1(predictions, labels)), "event_recall": recall, "mttd": float(detection_delay(predictions, labels))}


def run_corrected_analysis(score_root: Path | str, output_root: Path | str, datasets: Sequence[str] = ("SMD", "MSL", "SMAP", "PSM", "SWaT"), protocol: PrimaryProtocol = PRIMARY_PROTOCOL) -> dict[str, object]:
    root, output = Path(score_root), Path(output_root)
    paths = [p for dataset in datasets for p in sorted((root / dataset / "one_step").glob("*/scores.npz"))]
    runs, audit = load_unique_score_runs(paths)
    (output / "audits").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "audits" / "artifact_deduplication.json").write_text(__import__("json").dumps(audit, indent=2), encoding="utf-8")
    analyses = [analyze_dataset(run, training, protocol) for run, training in runs]
    frame = pd.concat(analyses, ignore_index=True) if analyses else pd.DataFrame()
    frame.to_csv(output / "metrics" / "event_level_rows.csv", index=False)
    frame.to_csv(output / "tables" / "delay_aware_performance.csv", index=False)
    frame[["dataset", "matched_pairs", "peak_smd", "tail_contrast", "contrast_ci_low", "contrast_ci_high", "regime", "evidence_status"]].drop_duplicates().to_csv(output / "tables" / "dataset_tail_contrast.csv", index=False)
    frame[["dataset", "matched_pairs", "peak_smd"]].drop_duplicates().to_csv(output / "tables" / "matching_balance.csv", index=False)
    return {"audit": audit, "rows": frame}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the fixed event-level SPL analysis.")
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["SMD", "MSL", "SMAP", "PSM", "SWaT"])
    args = parser.parse_args()
    run_corrected_analysis(args.score_root, args.output_dir, args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
