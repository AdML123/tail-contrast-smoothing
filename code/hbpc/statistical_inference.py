from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def _finite(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def rank_biserial(first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray) -> float:
    """Rank-biserial effect for second > first."""
    first_arr = _finite(first)
    second_arr = _finite(second)
    if first_arr.size == 0 or second_arr.size == 0:
        return float("nan")
    result = sp_stats.mannwhitneyu(second_arr, first_arr, alternative="two-sided")
    common_language = float(result.statistic) / float(first_arr.size * second_arr.size)
    return float(2.0 * common_language - 1.0)


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    arr = _finite(values)
    if arr.size == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        estimates[idx] = float(statistic(sample))
    alpha = 1.0 - float(confidence)
    return {
        "estimate": float(statistic(arr)),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "n": int(arr.size),
    }


def bootstrap_rank_biserial_ci(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    first_arr = _finite(first)
    second_arr = _finite(second)
    if first_arr.size == 0 or second_arr.size == 0:
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_first": int(first_arr.size),
            "n_second": int(second_arr.size),
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_boot), dtype=float)
    for idx in range(int(n_boot)):
        first_sample = rng.choice(first_arr, size=first_arr.size, replace=True)
        second_sample = rng.choice(second_arr, size=second_arr.size, replace=True)
        estimates[idx] = rank_biserial(first_sample, second_sample)
    alpha = 1.0 - float(confidence)
    return {
        "estimate": rank_biserial(first_arr, second_arr),
        "ci_low": float(np.quantile(estimates, alpha / 2.0)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha / 2.0)),
        "n_first": int(first_arr.size),
        "n_second": int(second_arr.size),
    }


def permutation_test_rank_biserial(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    alternative: str = "two-sided",
    max_exact: int = 200_000,
    n_permutations: int = 20_000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    first_arr = _finite(first)
    second_arr = _finite(second)
    observed = rank_biserial(first_arr, second_arr)
    n_first = first_arr.size
    n_second = second_arr.size
    if n_first == 0 or n_second == 0:
        return {"estimate": observed, "p_value": float("nan"), "n_permutations": 0, "mode": "empty"}
    pooled = np.concatenate([first_arr, second_arr])
    total_combinations = comb(pooled.size, n_second)
    if total_combinations <= int(max_exact):
        effects = []
        indices = range(pooled.size)
        for second_idx in combinations(indices, n_second):
            mask = np.zeros(pooled.size, dtype=bool)
            mask[list(second_idx)] = True
            effects.append(rank_biserial(pooled[~mask], pooled[mask]))
        mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        effects = []
        for _ in range(int(n_permutations)):
            permuted = rng.permutation(pooled)
            effects.append(rank_biserial(permuted[:n_first], permuted[n_first:]))
        mode = "monte_carlo"
    effects_arr = np.asarray(effects, dtype=float)
    if alternative == "greater":
        exceed = np.sum(effects_arr >= observed - 1e-12)
    elif alternative == "less":
        exceed = np.sum(effects_arr <= observed + 1e-12)
    elif alternative == "two-sided":
        exceed = np.sum(np.abs(effects_arr) >= abs(observed) - 1e-12)
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    p_value = (float(exceed) + 1.0) / (float(effects_arr.size) + 1.0)
    return {
        "estimate": float(observed),
        "p_value": float(p_value),
        "n_permutations": int(effects_arr.size),
        "mode": mode,
    }


def adjust_pvalues(p_values: Iterable[float], method: str = "holm") -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    method_norm = method.lower()
    if method_norm == "holm":
        running = 0.0
        n = ranked.size
        for idx, value in enumerate(ranked):
            running = max(running, float((n - idx) * value))
            adjusted[idx] = min(running, 1.0)
    elif method_norm in {"bh", "fdr_bh", "benjamini-hochberg"}:
        running = 1.0
        n = ranked.size
        for idx in range(n - 1, -1, -1):
            running = min(running, float(ranked[idx] * n / (idx + 1)))
            adjusted[idx] = min(running, 1.0)
    else:
        raise ValueError("method must be holm or bh")
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return [float(x) for x in out]


def leave_one_out_spearman(frame: pd.DataFrame, feature: str, target: str, group: str = "dataset") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(columns=["left_out", "n", "spearman_r", "p_value"])
    for left_out in sorted(frame[group].dropna().unique()):
        subset = frame[frame[group] != left_out]
        clean = subset[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if clean.shape[0] < 3:
            rho = float("nan")
            p_value = float("nan")
        else:
            rho, p_value = sp_stats.spearmanr(clean[feature], clean[target])
        rows.append({"left_out": left_out, "n": int(clean.shape[0]), "spearman_r": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)
