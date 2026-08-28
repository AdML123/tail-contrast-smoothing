import numpy as np


def quantile_threshold(scores: np.ndarray, calibration_fraction: float, quantile: float = 0.995) -> float:
    valid = np.asarray(scores, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        raise ValueError("Cannot fit threshold with no valid scores")
    count = max(1, int(len(valid) * calibration_fraction))
    calibration = valid[:count]
    return float(np.quantile(calibration, quantile))


def threshold_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return np.where(np.isnan(scores), 0, scores > threshold).astype(np.int64)


def spot_threshold(scores: np.ndarray, calibration_fraction: float, q: float) -> float:
    try:
        from ads_evt import SPOT
    except Exception:
        return quantile_threshold(scores, calibration_fraction, quantile=0.995)
    valid = np.asarray(scores, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    count = max(1, int(len(valid) * calibration_fraction))
    calibration = valid[:count]
    stream = valid[count:]
    if len(stream) == 0:
        return quantile_threshold(scores, calibration_fraction, quantile=0.995)
    detector = SPOT(q=q)
    detector.fit(init_data=calibration, data=stream)
    detector.initialize(level=0.98)
    result = detector.run(with_alarm=True)
    thresholds = result.get("thresholds", [])
    if thresholds:
        return float(thresholds[-1])
    return quantile_threshold(scores, calibration_fraction, quantile=0.995)
