import json
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hbpc.baselines import ar1_scores, moving_average_scores, var1_scores
from hbpc.data import TimeSeriesDataset
from hbpc.linear import parameter_count, train_linear_predictor
from hbpc.metrics import affiliation_f1, detection_delay, events_from_binary, f1, point_adjust
from hbpc.scoring import compute_horizon_scale, compute_median_iqr_reference, score_profiles
from hbpc.thresholds import quantile_threshold, threshold_scores
from hbpc.windows import make_training_windows, retrospective_errors


PROFILE_METHODS = {
    "hbpc_full",
    "hbpc_magnitude",
    "hbpc_shape",
    "multi_mean_raw",
    "multi_mean_norm",
    "multi_mean_norm_rms_clip",
    "multi_mean_norm_median_iqr",
    "hbpc_full_rms_clip",
    "hbpc_full_median_iqr",
}
RMS_CLIP_METHODS = {"multi_mean_norm_rms_clip", "hbpc_full_rms_clip"}
MEDIAN_IQR_METHODS = {"multi_mean_norm_median_iqr", "hbpc_full_median_iqr"}
SMD_GATE_METHODS = (
    "one_step",
    "multi_mean_raw",
    "multi_mean_norm_rms_clip",
    "multi_mean_norm_median_iqr",
    "hbpc_full_rms_clip",
    "hbpc_full_median_iqr",
)


def _linear_profile_scores(
    train: np.ndarray,
    test: np.ndarray,
    method: str,
    lookback: int,
    horizons: int,
    eta: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, int]:
    train_x, train_y = make_training_windows(train, lookback=lookback, horizons=horizons)
    test_x, _ = make_training_windows(test, lookback=lookback, horizons=horizons)
    model = train_linear_predictor(
        train_x,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    train_errors = retrospective_errors(train, train_pred, lookback=lookback, horizons=horizons)
    test_errors = retrospective_errors(test, test_pred, lookback=lookback, horizons=horizons)
    center = None
    if method in MEDIAN_IQR_METHODS:
        center, scale = compute_median_iqr_reference(train_errors)
    elif method in RMS_CLIP_METHODS:
        scale = compute_horizon_scale(train_errors, floor_percentile=10.0)
    else:
        scale = compute_horizon_scale(train_errors)
    scores = score_profiles(test_errors, scale, variant=method, eta=eta, center=center)
    params = parameter_count(channels=train.shape[1], lookback=lookback, horizons=horizons)
    return scores, params


def _score_single_horizon(errors: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        channel_scores = np.nanmean(errors, axis=1)
        if channel_scores.shape[1] == 1:
            return channel_scores[:, 0]
        return np.nanmean(channel_scores, axis=1) + np.nanmax(channel_scores, axis=1)


def predict_windows_streaming(model, series: np.ndarray, lookback: int, batch_size: int = 4096) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64)
    count = len(series) - int(lookback)
    if count <= 0:
        raise ValueError("series is shorter than lookback + 1")
    outputs: list[np.ndarray] = []
    for start in range(0, count, int(batch_size)):
        stop = min(count, start + int(batch_size))
        batch = np.empty((stop - start, int(lookback), series.shape[1]), dtype=np.float64)
        for offset, window_start in enumerate(range(start, stop)):
            batch[offset] = series[window_start : window_start + int(lookback)]
        outputs.append(model.predict(batch))
    return np.concatenate(outputs, axis=0)


def _one_step_scores(
    train: np.ndarray,
    test: np.ndarray,
    lookback: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    train_x, train_y = make_training_windows(train, lookback=lookback, horizons=1)
    model = train_linear_predictor(
        train_x,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    test_pred = predict_windows_streaming(model, test, lookback=lookback)
    test_errors = retrospective_errors(test, test_pred, lookback=lookback, horizons=1)
    scores = _score_single_horizon(test_errors)
    training_errors = retrospective_errors(train, model.predict(train_x), lookback=lookback, horizons=1)
    training_scores = _score_single_horizon(training_errors)
    params = parameter_count(channels=train.shape[1], lookback=lookback, horizons=1)
    return scores, training_scores, params


def _one_step_scores_segmented(
    train: np.ndarray,
    test_segments: tuple[np.ndarray, ...],
    lookback: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score independent test runs with one model while resetting each warmup."""
    train_x, train_y = make_training_windows(train, lookback=lookback, horizons=1)
    model = train_linear_predictor(
        train_x,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )
    segment_scores: list[np.ndarray] = []
    for segment in test_segments:
        test_pred = predict_windows_streaming(model, segment, lookback=lookback)
        test_errors = retrospective_errors(segment, test_pred, lookback=lookback, horizons=1)
        segment_scores.append(_score_single_horizon(test_errors))
    scores = np.concatenate(segment_scores)
    training_errors = retrospective_errors(train, model.predict(train_x), lookback=lookback, horizons=1)
    training_scores = _score_single_horizon(training_errors)
    params = parameter_count(channels=train.shape[1], lookback=lookback, horizons=1)
    return scores, training_scores, params


def run_one_method(
    dataset: TimeSeriesDataset,
    method: str,
    output_dir: Path,
    lookback: int,
    horizons: int,
    eta: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
    calibration_fraction: float,
) -> dict[str, object]:
    started = time.perf_counter()
    if method in PROFILE_METHODS:
        scores, params = _linear_profile_scores(
            train=dataset.train,
            test=dataset.test,
            method=method,
            lookback=lookback,
            horizons=horizons,
            eta=eta,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        training_scores = np.asarray(scores, dtype=float)
    elif method == "one_step":
        if dataset.test_segments is None:
            scores, training_scores, params = _one_step_scores(
                train=dataset.train,
                test=dataset.test,
                lookback=lookback,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
        else:
            scores, training_scores, params = _one_step_scores_segmented(
                train=dataset.train,
                test_segments=dataset.test_segments,
                lookback=lookback,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
    elif method == "moving_average":
        scores = moving_average_scores(dataset.test, window=lookback)
        training_scores = moving_average_scores(dataset.train, window=lookback)
        params = 0
    elif method == "ar1":
        scores = ar1_scores(dataset.train, dataset.test)
        training_scores = ar1_scores(dataset.train, dataset.train)
        params = 2 * dataset.channel_count
    elif method == "var1":
        scores = var1_scores(dataset.train, dataset.test, ridge=1e-3)
        training_scores = var1_scores(dataset.train, dataset.train, ridge=1e-3)
        params = dataset.channel_count * dataset.channel_count + dataset.channel_count
    else:
        raise KeyError(f"Unsupported method: {method}")

    threshold = quantile_threshold(scores, calibration_fraction=calibration_fraction, quantile=0.995)
    pred = threshold_scores(scores, threshold)
    labels = dataset.labels
    raw_f1 = f1(pred, labels)
    adjusted = point_adjust(pred, labels)
    pa_f1 = f1(adjusted, labels)
    aff_f1 = affiliation_f1(pred, labels)
    delay = detection_delay(pred, labels)
    pred_events = events_from_binary(pred)
    label_events = events_from_binary(labels)

    run_dir = Path(output_dir) / "raw" / dataset.name / method / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "scores": scores,
        "pred": pred,
        "labels": labels,
        "training_normal_scores": training_scores,
    }
    if dataset.test_segments is not None:
        payload["segment_lengths"] = np.asarray([len(segment) for segment in dataset.test_segments], dtype=np.int64)
    np.savez_compressed(run_dir / "scores.npz", **payload)
    metadata = {
        "dataset": dataset.name,
        "method": method,
        "seed": seed,
        "lookback": lookback,
        "horizons": horizons if method != "one_step" else 1,
        "eta": eta,
        "threshold": threshold,
        "calibration_fraction": calibration_fraction,
        "parameter_count": params,
        "elapsed_seconds": time.perf_counter() - started,
        "predicted_points": int(pred.sum()),
        "label_points": int(labels.sum()),
        "predicted_events": len(pred_events),
        "label_events": len(label_events),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset": dataset.name,
        "method": method,
        "seed": seed,
        "raw_f1": raw_f1,
        "pa_f1": pa_f1,
        "affiliation_f1": aff_f1,
        "delay": delay,
        "threshold": threshold,
        "parameter_count": params,
        "predicted_points": int(pred.sum()),
        "label_points": int(labels.sum()),
        "predicted_events": len(pred_events),
        "label_events": len(label_events),
    }


def write_metrics(rows: Iterable[dict[str, object]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)
    return output_path


def summarize_metrics(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        frame.groupby(["dataset", "method"], as_index=False)
        .agg(
            raw_f1_mean=("raw_f1", "mean"),
            raw_f1_std=("raw_f1", "std"),
            pa_f1_mean=("pa_f1", "mean"),
            pa_f1_std=("pa_f1", "std"),
            affiliation_f1_mean=("affiliation_f1", "mean"),
            affiliation_f1_std=("affiliation_f1", "std"),
            delay_mean=("delay", "mean"),
            delay_std=("delay", "std"),
            predicted_points_mean=("predicted_points", "mean"),
            label_points_mean=("label_points", "mean"),
            predicted_events_mean=("predicted_events", "mean"),
            label_events_mean=("label_events", "mean"),
        )
        .fillna(0.0)
    )
    summary.to_csv(output_path, index=False)
    return output_path


def write_smd_ablation_gate(frame: pd.DataFrame, output_path: Path) -> Path:
    methods = list(SMD_GATE_METHODS)
    subset = frame[(frame["dataset"] == "SMD") & (frame["method"].isin(methods))].copy()
    subset["method"] = pd.Categorical(subset["method"], categories=methods, ordered=True)
    subset = subset.sort_values(["method", "seed"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    return output_path
