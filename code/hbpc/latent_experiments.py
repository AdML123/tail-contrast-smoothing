import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hbpc.data import TimeSeriesDataset, load_dataset
from hbpc.experiments import run_one_method
from hbpc.latent import (
    LatentProfileConfig,
    count_latent_parameters,
    latent_profile_scores,
    train_latent_predictor,
)
from hbpc.metrics import affiliation_f1, detection_delay, events_from_binary, f1, point_adjust
from hbpc.thresholds import quantile_threshold, threshold_scores


BASELINE_METHODS = ("one_step", "multi_mean_raw", "multi_mean_norm_rms_clip")
LATENT_METHODS = ("latent_fixed_dct_single", "latent_fixed_dct_multi")


def _metric_row(
    dataset: TimeSeriesDataset,
    method: str,
    seed: int,
    scores: np.ndarray,
    parameter_count: int,
    threshold_quantile: float = 0.995,
    calibration_fraction: float = 0.10,
) -> dict[str, object]:
    threshold = quantile_threshold(scores, calibration_fraction=calibration_fraction, quantile=threshold_quantile)
    pred = threshold_scores(scores, threshold)
    adjusted = point_adjust(pred, dataset.labels)
    pred_events = events_from_binary(pred)
    label_events = events_from_binary(dataset.labels)
    return {
        "dataset": dataset.name,
        "method": method,
        "seed": seed,
        "raw_f1": f1(pred, dataset.labels),
        "pa_f1": f1(adjusted, dataset.labels),
        "affiliation_f1": affiliation_f1(pred, dataset.labels),
        "delay": detection_delay(pred, dataset.labels),
        "threshold": threshold,
        "parameter_count": parameter_count,
        "predicted_points": int(pred.sum()),
        "label_points": int(dataset.labels.sum()),
        "predicted_events": len(pred_events),
        "label_events": len(label_events),
    }


def _write_latent_raw(
    output_dir: Path,
    dataset: TimeSeriesDataset,
    method: str,
    seed: int,
    cfg: LatentProfileConfig,
    scores: np.ndarray,
    raw_scores: np.ndarray,
    parameter_count: int,
    elapsed_seconds: float,
) -> None:
    run_dir = Path(output_dir) / "raw" / dataset.name / method / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "scores.npz", scores=scores, raw_scores=raw_scores, labels=dataset.labels)
    metadata = {
        "dataset": dataset.name,
        "method": method,
        "seed": seed,
        "lookback": cfg.lookback,
        "patch_length": cfg.patch_length,
        "horizons": list(cfg.horizons),
        "latent_dim": cfg.latent_dim,
        "hidden_dim": cfg.hidden_dim,
        "epochs": cfg.epochs,
        "parameter_count": parameter_count,
        "elapsed_seconds": elapsed_seconds,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _run_latent_method(
    dataset: TimeSeriesDataset,
    output_dir: Path,
    method: str,
    cfg: LatentProfileConfig,
    seed: int,
    device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    method_cfg = replace(cfg, horizons=(1,)) if method == "latent_fixed_dct_single" else cfg
    model = train_latent_predictor(dataset.train, method_cfg, seed=seed, device=device)
    scores, raw_scores = latent_profile_scores(model, dataset.train, dataset.test, method_cfg, device=device)
    params = count_latent_parameters(model)
    _write_latent_raw(
        output_dir=output_dir,
        dataset=dataset,
        method=method,
        seed=seed,
        cfg=method_cfg,
        scores=scores,
        raw_scores=raw_scores,
        parameter_count=params,
        elapsed_seconds=time.perf_counter() - started,
    )
    return _metric_row(dataset, method, seed, scores, parameter_count=params)


def run_latent_pilot(
    dataset: TimeSeriesDataset,
    output_dir: Path,
    cfg: LatentProfileConfig,
    seed: int,
    device: str,
) -> Path:
    rows: list[dict[str, object]] = []
    for method in BASELINE_METHODS:
        rows.append(
            run_one_method(
                dataset=dataset,
                method=method,
                output_dir=output_dir,
                lookback=cfg.lookback,
                horizons=max(cfg.horizons),
                eta=1.0,
                epochs=cfg.epochs,
                learning_rate=cfg.learning_rate,
                seed=seed,
                device=device,
                calibration_fraction=0.10,
            )
        )
    for method in LATENT_METHODS:
        rows.append(_run_latent_method(dataset, output_dir, method, cfg, seed, device))

    output_path = Path(output_dir) / "metrics" / "latent_pilot_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _truncate_dataset(dataset: TimeSeriesDataset, max_train_points: int | None, max_test_points: int | None):
    train = dataset.train
    test = dataset.test
    labels = dataset.labels
    if max_train_points is not None:
        train = train[:max_train_points]
    if max_test_points is not None:
        test = test[:max_test_points]
        labels = labels[:max_test_points]
    return replace(dataset, train=train, test=test, labels=labels)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-target latent predictive profile pilot")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Time-Series-Library"))
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--output-dir", type=Path, default=Path("results-latent-smd-gate"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--patch-length", type=int, default=8)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-points", type=int, default=None)
    parser.add_argument("--max-test-points", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset = load_dataset(args.dataset_root, args.dataset)
    dataset = _truncate_dataset(dataset, args.max_train_points, args.max_test_points)
    cfg = LatentProfileConfig(
        lookback=args.lookback,
        patch_length=args.patch_length,
        horizons=tuple(args.horizons),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    rows = []
    for seed in args.seeds:
        path = run_latent_pilot(dataset, args.output_dir, cfg, seed=seed, device=args.device)
        rows.append(pd.read_csv(path))
    pd.concat(rows, ignore_index=True).to_csv(Path(args.output_dir) / "metrics" / "latent_pilot_metrics.csv", index=False)


if __name__ == "__main__":
    main()
