import numpy as np


def make_training_windows(series: np.ndarray, lookback: int, horizons: int) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float64)
    n, channels = series.shape
    count = n - lookback - horizons + 1
    if count <= 0:
        raise ValueError("series is shorter than lookback + horizons")
    x = np.empty((count, lookback, channels), dtype=np.float64)
    y = np.empty((count, horizons, channels), dtype=np.float64)
    for i in range(count):
        x[i] = series[i : i + lookback]
        y[i] = series[i + lookback : i + lookback + horizons]
    return x, y


def retrospective_errors(
    actual: np.ndarray,
    predictions: np.ndarray,
    lookback: int,
    horizons: int,
) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    n, _channels = actual.shape
    errors = np.full((n, horizons, actual.shape[1]), np.nan, dtype=np.float64)
    for window_index in range(predictions.shape[0]):
        window_end = window_index + lookback - 1
        for horizon_index in range(horizons):
            target_t = window_end + horizon_index + 1
            if target_t < n:
                errors[target_t, horizon_index, :] = np.abs(actual[target_t] - predictions[window_index, horizon_index])
    errors[: lookback + horizons - 1] = np.nan
    return errors
