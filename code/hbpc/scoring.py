import warnings

import numpy as np


DEFAULT_NORM_CLIP = 50.0


def compute_horizon_scale(
    train_errors: np.ndarray,
    eps: float = 1e-8,
    floor_percentile: float | None = None,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rms = np.sqrt(np.nanmean(train_errors**2, axis=0))
    scale = np.where(rms < eps, 1.0, rms)
    if floor_percentile is not None:
        valid = rms[np.isfinite(rms)]
        if valid.size:
            floor = max(float(np.percentile(valid, floor_percentile)), eps)
            scale = np.maximum(scale, floor)
    return scale


def compute_median_iqr_reference(
    train_errors: np.ndarray,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = np.nanmedian(train_errors, axis=0)
        q25 = np.nanpercentile(train_errors, 25.0, axis=0)
        q75 = np.nanpercentile(train_errors, 75.0, axis=0)
    iqr = q75 - q25
    positive = iqr[np.isfinite(iqr) & (iqr >= eps)]
    floor = float(np.percentile(positive, floor_percentile)) if positive.size else 1.0
    floor = max(floor, eps)
    scale = np.where(np.isfinite(iqr) & (iqr >= eps), iqr, floor)
    center = np.where(np.isfinite(center), center, 0.0)
    return center, scale


def _aggregate_channels(channel_scores: np.ndarray) -> np.ndarray:
    if channel_scores.shape[1] == 1:
        return channel_scores[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(channel_scores, axis=1) + np.nanmax(channel_scores, axis=1)


def _normalized_errors(
    errors: np.ndarray,
    scale: np.ndarray,
    center: np.ndarray | None,
    clip_max: float | None,
) -> np.ndarray:
    if center is None:
        normalized = errors / scale[None, :, :]
    else:
        normalized = (errors - center[None, :, :]) / scale[None, :, :]
        normalized = np.maximum(normalized, 0.0)
    if clip_max is not None:
        normalized = np.clip(normalized, 0.0, clip_max)
    return normalized


def score_profiles(
    errors: np.ndarray,
    scale: np.ndarray,
    variant: str,
    eta: float,
    center: np.ndarray | None = None,
    clip_max: float | None = DEFAULT_NORM_CLIP,
) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if variant == "multi_mean_raw":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            channel_scores = np.nanmean(errors, axis=1)
        return _aggregate_channels(channel_scores)

    if variant in {"multi_mean_norm_median_iqr", "hbpc_full_median_iqr"} and center is None:
        raise ValueError(f"{variant} requires a median/IQR center")

    clipped_variants = {
        "multi_mean_norm_rms_clip",
        "hbpc_full_rms_clip",
        "multi_mean_norm_median_iqr",
        "hbpc_full_median_iqr",
    }
    variant_clip = clip_max if variant in clipped_variants else None
    normalized = _normalized_errors(errors, scale, center=center, clip_max=variant_clip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(normalized, axis=1)
        std = np.nanstd(normalized, axis=1)

    if variant in {"hbpc_magnitude", "multi_mean_norm", "multi_mean_norm_rms_clip", "multi_mean_norm_median_iqr"}:
        channel_scores = mean
    elif variant == "hbpc_shape":
        channel_scores = std
    elif variant in {"hbpc_full", "hbpc_full_rms_clip", "hbpc_full_median_iqr"}:
        channel_scores = mean + eta * std
    else:
        raise KeyError(f"Unsupported scoring variant: {variant}")
    return _aggregate_channels(channel_scores)
