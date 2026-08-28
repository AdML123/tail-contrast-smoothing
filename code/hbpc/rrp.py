from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RelaxationFeatures:
    indices: np.ndarray
    peak: np.ndarray
    tail: np.ndarray
    relax: np.ndarray
    curves: np.ndarray


@dataclass(frozen=True)
class RankEffect:
    n_first: int
    n_second: int
    u: float
    common_language: float
    rank_biserial: float
    p_value: float
    median_first: float
    median_second: float
    median_ratio: float


def relaxation_features(scores: np.ndarray, k: int, eps: float = 1e-8) -> RelaxationFeatures:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size <= k:
        raise ValueError("scores length must be greater than k")

    indices = np.arange(arr.size - k, dtype=int)
    peak_raw = arr[: arr.size - k]
    peak = np.where(np.isfinite(peak_raw), peak_raw, 0.0)
    future = np.lib.stride_tricks.sliding_window_view(arr[1:], int(k))[: arr.size - k]
    finite_future = np.isfinite(future)
    totals = np.where(finite_future, future, 0.0).sum(axis=1)
    counts = finite_future.sum(axis=1)
    tail = np.divide(totals, counts, out=np.zeros_like(totals, dtype=float), where=counts > 0)
    relax = tail / (peak + eps)
    curves_window = np.lib.stride_tricks.sliding_window_view(arr, int(k) + 1)[: arr.size - k]
    curves = curves_window / (peak[:, None] + eps)
    curves = np.where(np.isfinite(curves), curves, 0.0)
    return RelaxationFeatures(
        indices=indices,
        peak=peak,
        tail=tail,
        relax=relax,
        curves=curves,
    )

def assign_rrp_groups(
    features: RelaxationFeatures,
    labels: np.ndarray,
    k: int,
    high_fraction: float = 0.01,
    typical_quantiles: tuple[float, float] = (0.4, 0.6),
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 < high_fraction < 1.0:
        raise ValueError("high_fraction must be in (0, 1)")
    low_q, high_q = typical_quantiles
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("typical_quantiles must be ordered within [0, 1]")

    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    if features.indices.size == 0 or labels_arr.size <= int(features.indices[-1]):
        raise ValueError("labels must cover feature indices")

    valid_labels = labels_arr[features.indices]
    groups = np.full(features.indices.shape, "other", dtype=object)
    normal_mask = ~valid_labels
    anomaly_mask = valid_labels
    finite_peak = np.isfinite(features.peak)
    normal_peak = features.peak[normal_mask & finite_peak]
    anomaly_peak = features.peak[anomaly_mask & finite_peak]

    if normal_peak.size:
        normal_high = float(np.quantile(normal_peak, 1.0 - high_fraction))
        normal_low = float(np.quantile(normal_peak, low_q))
        normal_typical_high = float(np.quantile(normal_peak, high_q))
        groups[normal_mask & (features.peak >= normal_high)] = "A"
        typical = normal_mask & (features.peak >= normal_low) & (features.peak <= normal_typical_high)
        groups[typical & (groups == "other")] = "C"

    if anomaly_peak.size:
        anomaly_high = float(np.quantile(anomaly_peak, 1.0 - high_fraction))
        event_lengths = anomaly_event_lengths(labels_arr)
        high_anomaly_positions = np.flatnonzero(anomaly_mask & (features.peak >= anomaly_high))
        for pos in high_anomaly_positions:
            idx = int(features.indices[pos])
            groups[pos] = "B1" if event_lengths[idx] <= k else "B2"

    return groups


def anomaly_event_lengths(labels: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels).astype(bool).reshape(-1)
    lengths = np.zeros(labels_arr.size, dtype=int)
    start: int | None = None
    for idx, value in enumerate(labels_arr):
        if value and start is None:
            start = idx
        if (not value) and start is not None:
            lengths[start:idx] = idx - start
            start = None
    if start is not None:
        lengths[start:] = labels_arr.size - start
    return lengths


def rank_effect(first: np.ndarray, second: np.ndarray) -> RankEffect:
    first_arr = _finite_values(first)
    second_arr = _finite_values(second)
    n_first = int(first_arr.size)
    n_second = int(second_arr.size)
    if n_first == 0 or n_second == 0:
        return RankEffect(
            n_first=n_first,
            n_second=n_second,
            u=np.nan,
            common_language=np.nan,
            rank_biserial=np.nan,
            p_value=np.nan,
            median_first=np.nan,
            median_second=np.nan,
            median_ratio=np.nan,
        )

    greater = float((second_arr[:, None] > first_arr[None, :]).sum())
    ties = float((second_arr[:, None] == first_arr[None, :]).sum())
    u = greater + 0.5 * ties
    common = u / float(n_first * n_second)
    median_first = float(np.median(first_arr))
    median_second = float(np.median(second_arr))
    median_ratio = float(median_second / median_first) if median_first != 0.0 else float("inf")
    return RankEffect(
        n_first=n_first,
        n_second=n_second,
        u=float(u),
        common_language=float(common),
        rank_biserial=float(2.0 * common - 1.0),
        p_value=_mann_whitney_p_value(first_arr, second_arr),
        median_first=median_first,
        median_second=median_second,
        median_ratio=median_ratio,
    )


def group_summary(
    groups: np.ndarray,
    relax: np.ndarray,
    peak: np.ndarray,
    tail: np.ndarray,
    k: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_arr = np.asarray(groups, dtype=object).reshape(-1)
    relax_arr = np.asarray(relax, dtype=float).reshape(-1)
    peak_arr = np.asarray(peak, dtype=float).reshape(-1)
    tail_arr = np.asarray(tail, dtype=float).reshape(-1)

    for group in ("A", "B", "B1", "B2", "C"):
        if group == "B":
            mask = (group_arr == "B1") | (group_arr == "B2")
        else:
            mask = group_arr == group
        values = _finite_values(relax_arr[mask])
        if values.size == 0:
            continue
        peaks = _finite_values(peak_arr[mask])
        tails = _finite_values(tail_arr[mask])
        rows.append(
            {
                "seed": seed,
                "k": k,
                "group": group,
                "count": int(values.size),
                "relax_mean": float(np.mean(values)),
                "relax_median": float(np.median(values)),
                "relax_q25": float(np.quantile(values, 0.25)),
                "relax_q75": float(np.quantile(values, 0.75)),
                "relax_q90": float(np.quantile(values, 0.90)),
                "peak_median": float(np.median(peaks)) if peaks.size else np.nan,
                "tail_median": float(np.median(tails)) if tails.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_phenomenon_gate(
    effects_by_k: dict[int, dict[str, RankEffect]],
    min_group_size: int = 5,
    min_rank_biserial: float = 0.3,
    min_median_ratio: float = 1.5,
    min_pass_horizons: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pass_count = 0
    for k in sorted(effects_by_k):
        effect = effects_by_k[k].get("A_vs_B2")
        if effect is None:
            continue
        gate_pass = bool(
            effect.n_first >= min_group_size
            and effect.n_second >= min_group_size
            and effect.rank_biserial >= min_rank_biserial
            and effect.median_ratio >= min_median_ratio
        )
        pass_count += int(gate_pass)
        rows.append(
            {
                "k": k,
                "comparison": "A_vs_B2",
                "n_A": effect.n_first,
                "n_B2": effect.n_second,
                "rank_biserial": effect.rank_biserial,
                "median_ratio": effect.median_ratio,
                "gate_pass": gate_pass,
            }
        )

    rows.append(
        {
            "k": "overall",
            "comparison": "A_vs_B2",
            "n_A": np.nan,
            "n_B2": np.nan,
            "rank_biserial": np.nan,
            "median_ratio": np.nan,
            "gate_pass": bool(pass_count >= min_pass_horizons),
        }
    )
    return pd.DataFrame(rows)


def _finite_mean(values: np.ndarray) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else 0.0


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _mann_whitney_p_value(first: np.ndarray, second: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(first, second, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def ewma_scores(scores: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr.copy()

    out = np.zeros(arr.size, dtype=float)
    out[0] = arr[0] if np.isfinite(arr[0]) else 0.0
    for idx in range(1, arr.size):
        value = arr[idx] if np.isfinite(arr[idx]) else 0.0
        out[idx] = alpha * value + (1.0 - alpha) * out[idx - 1]
    return out


def cusum_scores(
    scores: np.ndarray,
    reference: np.ndarray,
    drift: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    if drift < 0.0:
        raise ValueError("drift must be non-negative")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    ref = _finite_values(reference)
    if ref.size == 0:
        raise ValueError("reference must contain at least one finite value")

    median = float(np.median(ref))
    q75, q25 = np.percentile(ref, [75, 25])
    scale = max(float(q75 - q25), eps)
    safe = np.where(np.isfinite(arr), arr, median)
    z = np.maximum((safe - median) / scale, 0.0)
    out = np.zeros(arr.size, dtype=float)
    for idx, value in enumerate(z):
        previous = out[idx - 1] if idx else 0.0
        out[idx] = max(0.0, previous + float(value) - drift)
    return out


def tail_scores(scores: np.ndarray, k: int) -> np.ndarray:
    features = relaxation_features(scores, k=k)
    out = np.full(np.asarray(scores, dtype=float).reshape(-1).size, np.nan, dtype=float)
    out[features.indices] = features.tail
    return out


def peak_tail_scores(scores: np.ndarray, k: int) -> np.ndarray:
    features = relaxation_features(scores, k=k)
    out = np.full(np.asarray(scores, dtype=float).reshape(-1).size, np.nan, dtype=float)
    out[features.indices] = features.peak * features.tail
    return out


def peak_gated_tail_scores(scores: np.ndarray, k: int, peak_quantile: float) -> np.ndarray:
    if not 0.0 < peak_quantile < 1.0:
        raise ValueError("peak_quantile must be in (0, 1)")
    features = relaxation_features(scores, k=k)
    out = np.full(np.asarray(scores, dtype=float).reshape(-1).size, np.nan, dtype=float)
    valid_peak = features.peak[np.isfinite(features.peak)]
    if valid_peak.size == 0:
        return out

    threshold = float(np.quantile(valid_peak, peak_quantile))
    gated = np.where(features.peak >= threshold, features.tail, 0.0)
    out[features.indices] = gated
    return out


def candidate_count(scores: np.ndarray) -> int:
    arr = np.asarray(scores, dtype=float).reshape(-1)
    return int(np.logical_and(np.isfinite(arr), arr > 0.0).sum())


def top_n_alarms(scores: np.ndarray, top_n: int) -> np.ndarray:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    pred = np.zeros(arr.size, dtype=np.int64)
    candidate_indices = np.flatnonzero(np.isfinite(arr) & (arr > 0.0))
    if candidate_indices.size == 0:
        return pred

    ordered = sorted(candidate_indices.tolist(), key=lambda idx: (-arr[idx], idx))
    selected = ordered[: min(top_n, len(ordered))]
    pred[np.array(selected, dtype=int)] = 1
    return pred


def delayed_top_n_alarms(
    scores: np.ndarray,
    k: int,
    top_n: int,
    length: int | None = None,
) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=float).reshape(-1)
    output_length = arr.size if length is None else int(length)
    if output_length <= 0:
        raise ValueError("length must be positive")

    base = top_n_alarms(arr, top_n=top_n)
    pred = np.zeros(output_length, dtype=np.int64)
    alarm_times = np.flatnonzero(base) + k
    alarm_times = alarm_times[(alarm_times >= 0) & (alarm_times < output_length)]
    pred[alarm_times] = 1
    return pred
