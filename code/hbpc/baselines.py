import numpy as np
from sklearn.linear_model import Ridge


def _aggregate_abs_errors(errors: np.ndarray) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.shape[1] == 1:
        return errors[:, 0]
    return np.nanmean(errors, axis=1) + np.nanmax(errors, axis=1)


def moving_average_scores(test: np.ndarray, window: int) -> np.ndarray:
    test = np.asarray(test, dtype=np.float64)
    scores = np.full(len(test), np.nan, dtype=np.float64)
    for t in range(window, len(test)):
        pred = test[t - window : t].mean(axis=0)
        scores[t] = _aggregate_abs_errors(np.abs(test[t : t + 1] - pred))[0]
    return scores


def ar1_scores(train: np.ndarray, test: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    channels = train.shape[1]
    coef = np.empty(channels, dtype=np.float64)
    intercept = np.empty(channels, dtype=np.float64)
    for channel in range(channels):
        model = Ridge(alpha=ridge).fit(train[:-1, [channel]], train[1:, channel])
        coef[channel] = model.coef_[0]
        intercept[channel] = model.intercept_

    scores = np.full(len(test), np.nan, dtype=np.float64)
    for t in range(1, len(test)):
        pred = intercept + coef * test[t - 1]
        scores[t] = _aggregate_abs_errors(np.abs(test[t : t + 1] - pred))[0]
    return scores


def var1_scores(train: np.ndarray, test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    model = Ridge(alpha=ridge).fit(train[:-1], train[1:])
    scores = np.full(len(test), np.nan, dtype=np.float64)
    for t in range(1, len(test)):
        pred = model.predict(test[t - 1 : t])[0]
        scores[t] = _aggregate_abs_errors(np.abs(test[t : t + 1] - pred))[0]
    return scores
