from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ThresholdScanRow:
    tau: float
    p: float
    fae: float
    duty: float
    feasible: bool
    events: int
    alarm_points: int


@dataclass(frozen=True)
class CalibrationResult:
    calibrator: str
    tau: float
    tau_quantile: float
    fae: float
    duty: float
    plateau_lo: int | None = None
    plateau_hi: int | None = None
    w_q: float = 0.0
    m_rel: float = 0.0
    u: float = 0.0
    median_fae: float = 0.0
    delta_rel: float = 0.0
    stability: float = 0.0
    low_util_warning: bool = False


def eventize(alarms: np.ndarray, min_gap: int = 0) -> list[tuple[int, int]]:
    """Convert point alarms to inclusive event intervals."""
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    values = np.asarray(alarms).astype(bool)
    events: list[tuple[int, int]] = []
    start: int | None = None

    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            events.append((start, idx - 1))
            start = None

    if start is not None:
        events.append((start, len(values) - 1))

    if min_gap == 0 or len(events) <= 1:
        return events

    merged = [events[0]]
    for next_start, next_stop in events[1:]:
        prev_start, prev_stop = merged[-1]
        gap = next_start - prev_stop - 1
        if gap <= min_gap:
            merged[-1] = (prev_start, next_stop)
        else:
            merged.append((next_start, next_stop))
    return merged


def scan_thresholds(
    scores: np.ndarray,
    budget: float,
    gamma: float = 1.0,
    fs: float = 1.0,
    k: int = 1000,
    min_gap: int = 0,
) -> list[ThresholdScanRow]:
    """Sweep a quantile grid and compute nominal event-budget feasibility."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    score_arr = np.asarray(scores, dtype=float).reshape(-1)
    if score_arr.size == 0:
        raise ValueError("scores must not be empty")

    finite_scores = score_arr[np.isfinite(score_arr)]
    if finite_scores.size == 0:
        raise ValueError("scores must contain at least one finite value")

    probs = np.linspace(0.0, 1.0, k + 1, dtype=float)[1:]
    seen: dict[float, float] = {}
    for p in probs:
        tau = float(np.quantile(finite_scores, p))
        seen[tau] = max(seen.get(tau, 0.0), float(p))

    q_grid = np.array(sorted(seen.keys()), dtype=float)
    p_grid = np.array([seen[float(q)] for q in q_grid], dtype=float)

    hours = score_arr.size / (3600.0 * fs)
    safe_scores = np.where(np.isfinite(score_arr), score_arr, -np.inf)
    scan: list[ThresholdScanRow] = []

    for tau, p in zip(q_grid, p_grid):
        alarms = safe_scores > tau
        events = eventize(alarms, min_gap=min_gap)
        alarm_points = int(sum(stop - start + 1 for start, stop in events))
        fae = len(events) / hours if hours > 0 else 0.0
        duty = alarm_points / score_arr.size
        scan.append(
            ThresholdScanRow(
                tau=float(tau),
                p=float(p),
                fae=float(fae),
                duty=float(duty),
                feasible=bool(fae <= budget and duty <= gamma),
                events=len(events),
                alarm_points=alarm_points,
            )
        )

    return scan


def extract_plateaus(feasible: Sequence[bool]) -> list[tuple[int, int]]:
    plateaus: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(feasible):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            plateaus.append((start, idx - 1))
            start = None
    if start is not None:
        plateaus.append((start, len(feasible) - 1))
    return plateaus


def hn_lowest(scan: Sequence[ThresholdScanRow]) -> CalibrationResult:
    for row in scan:
        if row.feasible:
            return _row_result("hn_lowest", row)
    raise ValueError("no feasible threshold found")


def feasible_median(scan: Sequence[ThresholdScanRow]) -> CalibrationResult:
    feasible_indices = [idx for idx, row in enumerate(scan) if row.feasible]
    if not feasible_indices:
        raise ValueError("no feasible threshold found")
    row = scan[feasible_indices[(len(feasible_indices) - 1) // 2]]
    return _row_result("feasible_median", row)


def stable(
    scan: Sequence[ThresholdScanRow],
    budget: float,
    gamma: float = 1.0,
) -> CalibrationResult:
    return _stable_plateau_result(
        scan=scan,
        budget=budget,
        gamma=gamma,
        calibrator="stable",
        include_utilization=False,
        lambda_=0.0,
    )


def stable_u(
    scan: Sequence[ThresholdScanRow],
    budget: float,
    gamma: float = 1.0,
    lambda_: float = 0.3,
) -> CalibrationResult:
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    return _stable_plateau_result(
        scan=scan,
        budget=budget,
        gamma=gamma,
        calibrator="stable_u",
        include_utilization=True,
        lambda_=lambda_,
    )


def _row_result(calibrator: str, row: ThresholdScanRow) -> CalibrationResult:
    return CalibrationResult(
        calibrator=calibrator,
        tau=row.tau,
        tau_quantile=row.p,
        fae=row.fae,
        duty=row.duty,
        median_fae=row.fae,
    )


def _stable_plateau_result(
    scan: Sequence[ThresholdScanRow],
    budget: float,
    gamma: float,
    calibrator: str,
    include_utilization: bool,
    lambda_: float,
) -> CalibrationResult:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    plateaus = extract_plateaus([row.feasible for row in scan])
    if not plateaus:
        raise ValueError("no feasible threshold found")

    scored = [
        _score_plateau(scan, lo, hi, budget, gamma, include_utilization)
        for lo, hi in plateaus
    ]

    low_util_warning = False
    candidates = scored
    if include_utilization:
        candidates = [
            item for item in scored if item["median_fae"] >= lambda_ * budget
        ]
        if not candidates:
            candidates = scored
            low_util_warning = True

    best = max(candidates, key=lambda item: item["stability"])
    mid_idx = (int(best["lo"]) + int(best["hi"])) // 2
    row = scan[mid_idx]
    if not row.feasible:
        raise RuntimeError("selected an infeasible threshold outside its plateau")

    return CalibrationResult(
        calibrator=calibrator,
        tau=row.tau,
        tau_quantile=row.p,
        fae=row.fae,
        duty=row.duty,
        plateau_lo=int(best["lo"]),
        plateau_hi=int(best["hi"]),
        w_q=float(best["w_q"]),
        m_rel=float(best["m_rel"]),
        u=float(best["u"]),
        median_fae=float(best["median_fae"]),
        delta_rel=float(best["delta_rel"]),
        stability=float(best["stability"]),
        low_util_warning=low_util_warning,
    )


def _score_plateau(
    scan: Sequence[ThresholdScanRow],
    lo: int,
    hi: int,
    budget: float,
    gamma: float,
    include_utilization: bool,
) -> dict[str, float]:
    rows = scan[lo : hi + 1]
    faes = np.array([row.fae for row in rows], dtype=float)
    duties = np.array([row.duty for row in rows], dtype=float)

    p_left = scan[lo - 1].p if lo > 0 else 0.0
    p_right = scan[hi].p
    w_q = max(float(p_right - p_left), 0.0)
    margin_fae = (budget - faes) / budget
    margin_duty = (gamma - duties) / gamma
    m_rel = float(np.min(np.minimum(margin_fae, margin_duty)))
    median_fae = float(np.median(faes))
    u = min(median_fae / budget, 1.0)
    delta_rel = float((np.max(faes) - np.min(faes)) / budget)
    utilization = u if include_utilization else 1.0
    stability = w_q * m_rel * utilization / (1.0 + delta_rel)

    return {
        "lo": float(lo),
        "hi": float(hi),
        "w_q": w_q,
        "m_rel": m_rel,
        "u": u,
        "median_fae": median_fae,
        "delta_rel": delta_rel,
        "stability": float(stability),
    }
