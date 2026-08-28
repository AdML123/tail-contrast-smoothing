from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROUND_TOL = 5e-4

EXPECTED: dict[str, list[dict[str, object]]] = {
    "phenomenon": [
        {"dataset": "SMD", "k": 3, "n_A": 1485, "n_B": 18, "rank_biserial": 0.978},
        {"dataset": "SMD", "k": 5, "n_A": 1485, "n_B": 18, "rank_biserial": 0.939},
        {"dataset": "SMD", "k": 10, "n_A": 1485, "n_B": 18, "rank_biserial": 0.533},
        {"dataset": "SMD", "k": 20, "n_A": 1485, "n_B": 18, "rank_biserial": -0.174},
        {"dataset": "MSL", "k": 3, "n_A": 1296, "n_B": 207, "rank_biserial": 0.619},
        {"dataset": "MSL", "k": 5, "n_A": 1296, "n_B": 207, "rank_biserial": 0.621},
        {"dataset": "MSL", "k": 10, "n_A": 1296, "n_B": 207, "rank_biserial": 0.646},
        {"dataset": "MSL", "k": 20, "n_A": 1296, "n_B": 207, "rank_biserial": 0.649},
        {"dataset": "SMAP", "k": 3, "n_A": 1428, "n_B": 72, "rank_biserial": 0.255},
        {"dataset": "SMAP", "k": 5, "n_A": 1428, "n_B": 72, "rank_biserial": 0.126},
        {"dataset": "SMAP", "k": 10, "n_A": 1428, "n_B": 72, "rank_biserial": 0.237},
        {"dataset": "SMAP", "k": 20, "n_A": 1428, "n_B": 72, "rank_biserial": 0.310},
        {"dataset": "PSM", "k": 3, "n_A": 1134, "n_B": 369, "rank_biserial": 0.082},
        {"dataset": "PSM", "k": 5, "n_A": 1134, "n_B": 369, "rank_biserial": 0.070},
        {"dataset": "PSM", "k": 10, "n_A": 1134, "n_B": 369, "rank_biserial": 0.072},
        {"dataset": "PSM", "k": 20, "n_A": 1134, "n_B": 369, "rank_biserial": 0.135},
        {"dataset": "SWaT", "k": 3, "n_A": 1368, "n_B": 135, "rank_biserial": -0.302},
        {"dataset": "SWaT", "k": 5, "n_A": 1368, "n_B": 135, "rank_biserial": -0.294},
        {"dataset": "SWaT", "k": 10, "n_A": 1368, "n_B": 135, "rank_biserial": -0.374},
        {"dataset": "SWaT", "k": 20, "n_A": 1368, "n_B": 135, "rank_biserial": -0.161},
    ],
    "fixed_budget": [
        {"dataset": "SMD", "postprocess": "raw", "raw_f1": 0.122, "pa_f1": 0.687, "event_recall": 0.643, "event_precision": 0.071, "mttd": 15.56},
        {"dataset": "SMD", "postprocess": "ewma", "raw_f1": 0.362, "pa_f1": 0.861, "event_recall": 0.786, "event_precision": 0.333, "mttd": 2.50},
        {"dataset": "SMD", "postprocess": "forward_avg", "k": 3, "raw_f1": 0.371, "pa_f1": 0.867, "event_recall": 0.821, "event_precision": 0.422, "mttd": 2.35},
        {"dataset": "MSL", "postprocess": "raw", "raw_f1": 0.032, "pa_f1": 0.707, "event_recall": 0.417, "event_precision": 0.229, "mttd": 134.00},
        {"dataset": "MSL", "postprocess": "ewma", "raw_f1": 0.036, "pa_f1": 0.090, "event_recall": 0.125, "event_precision": 0.241, "mttd": 9.33},
        {"dataset": "MSL", "postprocess": "forward_avg", "k": 3, "raw_f1": 0.035, "pa_f1": 0.116, "event_recall": 0.167, "event_precision": 0.222, "mttd": 20.50},
        {"dataset": "SMAP", "postprocess": "raw", "raw_f1": 0.001, "pa_f1": 0.061, "event_recall": 0.143, "event_precision": 0.004, "mttd": 24.00},
        {"dataset": "SMAP", "postprocess": "ewma", "raw_f1": 0.000, "pa_f1": 0.000, "event_recall": 0.000, "event_precision": 0.000},
        {"dataset": "SMAP", "postprocess": "forward_avg", "k": 3, "raw_f1": 0.000, "pa_f1": 0.000, "event_recall": 0.000, "event_precision": 0.000},
        {"dataset": "PSM", "postprocess": "raw", "raw_f1": 0.036, "pa_f1": 0.939, "event_recall": 0.574, "event_precision": 0.690, "mttd": 136.85},
        {"dataset": "PSM", "postprocess": "ewma", "raw_f1": 0.039, "pa_f1": 0.940, "event_recall": 0.553, "event_precision": 0.932, "mttd": 194.15},
        {"dataset": "PSM", "postprocess": "forward_avg", "k": 3, "raw_f1": 0.037, "pa_f1": 0.939, "event_recall": 0.553, "event_precision": 0.880, "mttd": 194.04},
        {"dataset": "SWaT", "postprocess": "raw", "raw_f1": 0.070, "pa_f1": 0.671, "event_recall": 0.500, "event_precision": 0.567, "mttd": 95.25},
        {"dataset": "SWaT", "postprocess": "ewma", "raw_f1": 0.068, "pa_f1": 0.671, "event_recall": 0.500, "event_precision": 0.500, "mttd": 61.25},
        {"dataset": "SWaT", "postprocess": "forward_avg", "k": 3, "raw_f1": 0.068, "pa_f1": 0.671, "event_recall": 0.500, "event_precision": 0.424, "mttd": 58.50},
    ],
    "delay_fair": [
        {"postprocess": "raw_delayed", "k": 3, "raw_f1": 0.094, "pa_f1": 0.652, "event_recall": 0.250, "event_precision": 0.026, "event_f1": 0.047, "mttd": 42.29},
        {"postprocess": "ewma_delayed", "k": 3, "raw_f1": 0.296, "pa_f1": 0.810, "event_recall": 0.393, "event_precision": 0.192, "event_f1": 0.258, "mttd": 7.82},
        {"postprocess": "forward_avg", "k": 3, "raw_f1": 0.371, "pa_f1": 0.867, "event_recall": 0.821, "event_precision": 0.422, "event_f1": 0.557, "mttd": 2.35},
    ],
    "adaptation": [
        {"dataset": "SMD", "tau_anomaly_median": 0.74, "tau_normal_median": 0.008, "tau_ratio": 91.3},
        {"dataset": "MSL", "tau_anomaly_median": 17.50, "tau_normal_median": 0.024, "tau_ratio": 738.9},
        {"dataset": "SMAP", "tau_anomaly_median": 11.36, "tau_normal_median": 1.356, "tau_ratio": 8.4},
        {"dataset": "PSM", "tau_anomaly_median": 12.88, "tau_normal_median": 0.364, "tau_ratio": 35.4},
        {"dataset": "SWaT", "tau_anomaly_median": 82.64, "tau_normal_median": 4.131, "tau_ratio": 20.0},
    ],
    "correlation": [
        {"feature": "tau_anomaly_median", "target": "r_K3", "spearman_r": -0.700},
        {"feature": "tau_normal_median", "target": "r_K3", "spearman_r": -0.900, "spearman_p": 0.037},
        {"feature": "tau_ratio", "target": "r_K3", "spearman_r": 0.600},
    ],
}

PUBLIC_DEEP_EXPECTED = [
    {"predictor": "AnomalyTransformer", "postprocess": "raw", "top_n": 300, "k": 0, "raw_f1_mean": 0.1987480438184663, "raw_f1_std": 0.0870234789063348, "pa_f1_mean": 0.7423575984851802, "pa_f1_std": 0.04850513162284971, "event_recall_mean": 0.5595238095238094, "event_recall_std": 0.16877912950902177, "mttd_mean": 1.8888888888888886},
    {"predictor": "AnomalyTransformer", "postprocess": "forward_avg", "top_n": 500, "k": 20, "raw_f1_mean": 0.3669201520912548, "raw_f1_std": 0.08092698347449717, "pa_f1_mean": 0.6516690157618048, "pa_f1_std": 0.0383485998213566, "event_recall_mean": 0.25, "event_recall_std": 0.03571428571428576, "mttd_mean": 5.527777777777778},
    {"predictor": "Autoformer", "postprocess": "raw", "top_n": 300, "k": 0, "raw_f1_mean": 0.6674491392801252, "raw_f1_std": 0.0013552823220414267, "pa_f1_mean": 0.8631510602118485, "pa_f1_std": 0.0004991724510496922, "event_recall_mean": 0.2857142857142857, "event_recall_std": 0.0, "mttd_mean": 4.375},
    {"predictor": "Autoformer", "postprocess": "forward_avg", "top_n": 300, "k": 5, "raw_f1_mean": 0.6948356807511736, "raw_f1_std": 0.0, "pa_f1_mean": 0.8488612836438922, "pa_f1_std": 0.0, "event_recall_mean": 0.25, "event_recall_std": 0.0, "mttd_mean": 5.857142857142857},
    {"predictor": "TimesNet", "postprocess": "raw", "top_n": 500, "k": 0, "raw_f1_mean": 0.4892268694550064, "raw_f1_std": 0.03727076287080515, "pa_f1_mean": 0.7725180008296634, "pa_f1_std": 0.011730289656921213, "event_recall_mean": 0.7857142857142857, "event_recall_std": 0.0, "mttd_mean": 1.0606060606060608},
    {"predictor": "TimesNet", "postprocess": "forward_avg", "top_n": 500, "k": 2, "raw_f1_mean": 0.5095057034220533, "raw_f1_std": 0.03701125918998445, "pa_f1_mean": 0.7661005222229479, "pa_f1_std": 0.009458951866241485, "event_recall_mean": 0.6666666666666666, "event_recall_std": 0.04123930494211609, "mttd_mean": 1.1611111111111112},
    {"predictor": "Transformer", "postprocess": "raw", "top_n": 300, "k": 0, "raw_f1_mean": 0.6666666666666666, "raw_f1_std": 0.0, "pa_f1_mean": 0.8628628628628631, "pa_f1_std": 0.0, "event_recall_mean": 0.2857142857142857, "event_recall_std": 0.0, "mttd_mean": 4.375},
    {"predictor": "Transformer", "postprocess": "forward_avg", "top_n": 300, "k": 5, "raw_f1_mean": 0.6948356807511736, "raw_f1_std": 0.0, "pa_f1_mean": 0.8488612836438922, "pa_f1_std": 0.0, "event_recall_mean": 0.25, "event_recall_std": 0.0, "mttd_mean": 5.857142857142857},
]

NEGATIVE_EXPECTED = [
    {"source": "rrp", "family": "tail_K", "raw_f1": 0.371, "pa_f1": 0.867, "event_recall": 0.821, "mttd": 2.35},
    {"source": "rrp", "family": "peak_gated_tail_K", "raw_f1": 0.230, "pa_f1": 0.799, "event_recall": 0.464, "mttd": 6.08},
    {"source": "mars", "family": "mars_rel", "raw_f1": 0.371, "pa_f1": 0.858, "event_recall": 0.714, "mttd": 2.70},
    {"source": "ceres", "family": "deployment_ceres_envelope", "raw_f1": 0.317, "pa_f1": 0.824, "event_recall": 0.464, "mttd": 4.85},
    {"source": "ceres", "family": "oracle_ceres_lite", "raw_f1": 0.333, "pa_f1": 0.840, "event_recall": 0.607, "mttd": 3.35},
    {"source": "scar", "family": "deployment_scar", "raw_f1": 0.371, "pa_f1": 0.864, "event_recall": 0.786, "mttd": 2.41},
]

MAJOR_REVISION_EXPECTED: dict[str, list[dict[str, object]]] = {
    "synthetic": [
        {"noise_family": "gaussian", "phi": 0.3, "mu_normal": 0.2, "mu_anomaly": 2.0, "delta_mu": 1.8, "raw_f1_raw": 0.09540229885057472, "forward_f1_raw": 0.20344827586206897, "forward_gain": 0.10804597701149429, "rank_biserial": 1.0},
        {"noise_family": "gaussian", "phi": 0.3, "mu_normal": 1.4, "mu_anomaly": 1.4, "delta_mu": 0.0, "raw_f1_raw": 0.09540229885057472, "forward_f1_raw": 0.10000000000000001, "forward_gain": 0.00459770114942529, "rank_biserial": -0.05},
        {"noise_family": "gaussian", "phi": 0.3, "mu_normal": 2.0, "mu_anomaly": 0.2, "delta_mu": -1.8, "raw_f1_raw": 0.09540229885057472, "forward_f1_raw": 0.0011494252873563218, "forward_gain": -0.09425287356321839, "rank_biserial": -1.0},
    ],
    "synthetic_delta": [
        {"regime": "positive", "mean_forward_gain": 0.0646445295870583, "mean_raw_f1": 0.09540229885057468, "mean_rank_biserial": 0.9920833333333334, "n": 54},
        {"regime": "near_equal", "mean_forward_gain": 0.0016922094508301203, "mean_raw_f1": 0.09540229885057468, "mean_rank_biserial": -0.01906249999999997, "n": 36},
        {"regime": "negative", "mean_forward_gain": -0.05172413793103447, "mean_raw_f1": 0.09540229885057468, "mean_rank_biserial": -0.9946064814814813, "n": 54},
    ],
    "uncertainty": [
        {"dataset": "SMD", "predictor": "one_step", "k": 3, "n_A": 1485, "n_B": 18, "rank_biserial": 0.9784511784511785, "ci_low": 0.9553273475495697, "ci_high": 0.9975364758698092, "p_perm": 0.00004999750012499375, "permutation_mode": "monte_carlo", "p_holm": 0.000999950002499875, "p_bh": 0.00009090454568180681},
        {"dataset": "MSL", "predictor": "one_step", "k": 3, "n_A": 1296, "n_B": 207, "rank_biserial": 0.6194981213097155, "ci_low": 0.5519672198365837, "ci_high": 0.6841850808135027, "p_perm": 0.00004999750012499375, "permutation_mode": "monte_carlo", "p_holm": 0.000999950002499875, "p_bh": 0.00009090454568180681},
        {"dataset": "SMAP", "predictor": "one_step", "k": 3, "n_A": 1428, "n_B": 72, "rank_biserial": 0.25507713025210085, "ci_low": 0.15188193076758762, "ci_high": 0.3644095231903394, "p_perm": 0.00034998250087495626, "permutation_mode": "monte_carlo", "p_holm": 0.00279986000699965, "p_bh": 0.0005384346167307019},
        {"dataset": "PSM", "predictor": "one_step", "k": 3, "n_A": 1134, "n_B": 369, "rank_biserial": 0.08168792569659437, "ci_low": 0.002066969397405202, "ci_high": 0.15859326309459064, "p_perm": 0.017299135043247838, "permutation_mode": "monte_carlo", "p_holm": 0.08649567521623919, "p_bh": 0.021623918804059463},
        {"dataset": "SWaT", "predictor": "one_step", "k": 3, "n_A": 1368, "n_B": 135, "rank_biserial": -0.3018518518518518, "ci_low": -0.4061728395061728, "ci_high": -0.2018480949009287, "p_perm": 0.00004999750012499375, "permutation_mode": "monte_carlo", "p_holm": 0.000999950002499875, "p_bh": 0.00009090454568180681},
    ],
    "tau_uncertainty": [
        {"dataset": "SMD", "predictor": "one_step", "metric": "tau_anomaly_median", "estimate": 0.7384624969221785, "ci_low": 0.7384624969221785, "ci_high": 0.7384624969221785, "n": 3},
        {"dataset": "SMD", "predictor": "one_step", "metric": "tau_normal_median", "estimate": 0.008085403982021, "ci_low": 0.008085403982021, "ci_high": 0.008085403982021, "n": 3},
        {"dataset": "SMD", "predictor": "one_step", "metric": "tau_ratio", "estimate": 91.33278913017135, "ci_low": 91.33278913017135, "ci_high": 91.33278913017135, "n": 3},
        {"dataset": "MSL", "predictor": "one_step", "metric": "tau_anomaly_median", "estimate": 17.50172317978459, "ci_low": 17.50172317978459, "ci_high": 17.50172317978459, "n": 3},
        {"dataset": "MSL", "predictor": "one_step", "metric": "tau_normal_median", "estimate": 0.0236872294557508, "ci_low": 0.0236872294557508, "ci_high": 0.0236872294557508, "n": 3},
        {"dataset": "MSL", "predictor": "one_step", "metric": "tau_ratio", "estimate": 738.8674649552783, "ci_low": 738.8674649552783, "ci_high": 738.8674649552783, "n": 3},
        {"dataset": "SMAP", "predictor": "one_step", "metric": "tau_anomaly_median", "estimate": 11.355613695403372, "ci_low": 11.355613695403372, "ci_high": 11.355613695403372, "n": 3},
        {"dataset": "SMAP", "predictor": "one_step", "metric": "tau_normal_median", "estimate": 1.3561037070344732, "ci_low": 1.3561037070344732, "ci_high": 1.3561037070344732, "n": 3},
        {"dataset": "SMAP", "predictor": "one_step", "metric": "tau_ratio", "estimate": 8.373705961055014, "ci_low": 8.373705961055014, "ci_high": 8.373705961055014, "n": 3},
        {"dataset": "PSM", "predictor": "one_step", "metric": "tau_anomaly_median", "estimate": 12.881308554743056, "ci_low": 12.881308554743056, "ci_high": 12.881308554743056, "n": 3},
        {"dataset": "PSM", "predictor": "one_step", "metric": "tau_normal_median", "estimate": 0.3639739991238078, "ci_low": 0.3639739991238078, "ci_high": 0.3639739991238078, "n": 3},
        {"dataset": "PSM", "predictor": "one_step", "metric": "tau_ratio", "estimate": 35.390738310297294, "ci_low": 35.390738310297294, "ci_high": 35.390738310297294, "n": 3},
        {"dataset": "SWaT", "predictor": "one_step", "metric": "tau_anomaly_median", "estimate": 82.63827242570004, "ci_low": 82.63827242570004, "ci_high": 82.63827242570004, "n": 3},
        {"dataset": "SWaT", "predictor": "one_step", "metric": "tau_normal_median", "estimate": 4.131331934118921, "ci_low": 4.131331934118921, "ci_high": 4.131331934118921, "n": 3},
        {"dataset": "SWaT", "predictor": "one_step", "metric": "tau_ratio", "estimate": 20.00281597884342, "ci_low": 20.00281597884342, "ci_high": 20.00281597884342, "n": 3},
    ],
    "loo": [
        {"left_out": "MSL", "n": 12, "spearman_r": -0.8, "p_value": 0.001781840000000003},
        {"left_out": "PSM", "n": 12, "spearman_r": -1.0, "p_value": 0.0},
        {"left_out": "SMAP", "n": 12, "spearman_r": -1.0, "p_value": 0.0},
        {"left_out": "SMD", "n": 12, "spearman_r": -0.8, "p_value": 0.001781840000000003},
        {"left_out": "SWaT", "n": 12, "spearman_r": -0.8, "p_value": 0.001781840000000003},
    ],
    "sensitivity": [
        {"split_scope": "capped", "dataset": "MSL", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.05869648075212806, "best_forward_raw_f1": 0.08664718587218904, "best_ewma_raw_f1": 0.0774996823783509, "best_forward_gain": 0.027950705120060988},
        {"split_scope": "capped", "dataset": "PSM", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.11660483077156053, "best_forward_raw_f1": 0.11857348375861287, "best_ewma_raw_f1": 0.12069357159082304, "best_forward_gain": 0.001968652987052344},
        {"split_scope": "capped", "dataset": "SMAP", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.005294117647058823, "best_forward_raw_f1": 0.005294117647058823, "best_ewma_raw_f1": 0.001764705882352941, "best_forward_gain": 0.0},
        {"split_scope": "capped", "dataset": "SMD", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.14068441064638784, "best_forward_raw_f1": 0.37089201877934275, "best_ewma_raw_f1": 0.3615023474178404, "best_forward_gain": 0.2302076081329549},
        {"split_scope": "capped", "dataset": "SWaT", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.150384193194292, "best_forward_raw_f1": 0.15916575192096596, "best_ewma_raw_f1": 0.15184778631540433, "best_forward_gain": 0.008781558726673966},
        {"split_scope": "full", "dataset": "MSL", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.05042208532968286, "best_forward_raw_f1": 0.08464522016883413, "best_ewma_raw_f1": 0.06913073237508555, "best_forward_gain": 0.03422313483915127},
        {"split_scope": "full", "dataset": "PSM", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.05563216579330995, "best_forward_raw_f1": 0.058390134352468384, "best_ewma_raw_f1": 0.05815373704739766, "best_forward_gain": 0.0027579685591584344},
        {"split_scope": "full", "dataset": "SMAP", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.0034472852628555013, "best_forward_raw_f1": 0.01425596093076702, "best_ewma_raw_f1": 0.007505027291008331, "best_forward_gain": 0.01080867566791152},
        {"split_scope": "full", "dataset": "SMD", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.04086191039285245, "best_forward_raw_f1": 0.05058468006832217, "best_ewma_raw_f1": 0.05439495467087111, "best_forward_gain": 0.009722769675469714},
        {"split_scope": "full", "dataset": "SWaT", "predictor": "one_step", "normalization": "raw", "best_raw_raw_f1": 0.01582136243505151, "best_forward_raw_f1": 0.01582136243505151, "best_ewma_raw_f1": 0.016216896495927797, "best_forward_gain": 0.0},
    ],
    "swat_highpass": [
        {"dataset": "SWaT", "predictor": "one_step", "seed": 0, "filter": "none", "filter_window": 0, "tau_anomaly_median": 82.63827242570004, "tau_normal_median": 4.131331934118921, "tau_ratio": 20.00281597884342, "raw_best_f1": 0.06966009232060429, "forward_best_f1": 0.06840117498950904, "forward_gain": -0.0012589173310952467, "r_K3": -0.3042246642246642, "r_K5": -0.2930891330891331},
        {"dataset": "SWaT", "predictor": "one_step", "seed": 0, "filter": "highpass", "filter_window": 101, "tau_anomaly_median": 19.728219339134483, "tau_normal_median": 5.971791632743843, "tau_ratio": 3.303567932772968, "raw_best_f1": 0.038187159043222826, "forward_best_f1": 0.039026437263953, "forward_gain": 0.000839278220730176, "r_K3": -0.12371184371184374, "r_K5": -0.12273504273504277},
        {"dataset": "SWaT", "predictor": "one_step", "seed": 0, "filter": "highpass", "filter_window": 501, "tau_anomaly_median": 65.63825881889478, "tau_normal_median": 66.21329828435111, "tau_ratio": 0.9913153478174908, "raw_best_f1": 0.04951741502308015, "forward_best_f1": 0.04699958036088963, "forward_gain": -0.002517834662190521, "r_K3": -0.3772893772893773, "r_K5": -0.37181929181929185},
        {"dataset": "SWaT", "predictor": "one_step", "seed": 0, "filter": "highpass", "filter_window": 1001, "tau_anomaly_median": 70.58139745144169, "tau_normal_median": 5.523505917391955, "tau_ratio": 12.778369120453164, "raw_best_f1": 0.05203524968527067, "forward_best_f1": 0.049937054133445236, "forward_gain": -0.0020981955518254367, "r_K3": -0.3366544566544567, "r_K5": -0.32219780219780225},
    ],
    "public_deep": PUBLIC_DEEP_EXPECTED,
}


def verify_paper_results(results_root: Path | str, required_groups: Iterable[str] | None = None) -> dict[str, object]:
    root = Path(results_root)
    required = set(required_groups or EXPECTED.keys())
    failures: list[str] = []
    warnings: list[str] = []
    checked = 0

    table_specs = {
        "phenomenon": (root / "results-strr-five-dataset" / "tables" / "cross_dataset_phenomenon.csv", ("dataset", "k"), {"seed": -1}),
        "fixed_budget": (root / "results-strr-five-dataset" / "tables" / "budget_curve_summary.csv", ("dataset", "postprocess", "k"), {"top_n": 300}),
        "delay_fair": (root / "results-strr-five-dataset" / "tables" / "delay_fairness.csv", ("postprocess", "k"), {"dataset": "SMD", "top_n": 300}),
        "adaptation": (root / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_summary.csv", ("dataset",), {}),
        "correlation": (root / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_correlation_summary.csv", ("feature", "target"), {}),
    }

    for group, expected_rows in EXPECTED.items():
        if group not in required:
            continue
        path, keys, filters = table_specs[group]
        if not path.exists():
            failures.append(f"{group}: missing table {path}")
            continue
        frame = pd.read_csv(path)
        for col, value in filters.items():
            frame = frame[frame[col].astype(str) == str(value)]
        checked += _verify_rows(group, frame, keys, expected_rows, failures)


    if "negative" in required:
        negative_tables = {
            "rrp": root / "results-rrp-detector-pilot" / "tables" / "rrp_detector_best.csv",
            "mars": root / "results-mars-smd-pilot" / "tables" / "mars_best.csv",
            "ceres": root / "results-ceres-smd-pilot" / "tables" / "ceres_best_by_family.csv",
            "scar": root / "results-scar-smd-pilot" / "tables" / "scar_best_by_family.csv",
        }
        frames = []
        for source, path in negative_tables.items():
            if not path.exists():
                failures.append(f"negative: missing table {path}")
                continue
            frame = pd.read_csv(path)
            frame["source"] = source
            frames.append(frame)
        if frames:
            checked += _verify_rows("negative", pd.concat(frames, ignore_index=True), ("source", "family"), NEGATIVE_EXPECTED, failures)

    if "synthetic" in required:
        checked += _verify_major_revision_table(
            "synthetic",
            root / "results-synthetic-regime" / "tables" / "synthetic_regime_summary.csv",
            ("noise_family", "phi", "mu_normal", "mu_anomaly"),
            MAJOR_REVISION_EXPECTED["synthetic"],
            failures,
        )
        checked += _verify_major_revision_table(
            "synthetic_delta",
            root / "results-synthetic-regime" / "tables" / "synthetic_delta_summary.csv",
            ("regime",),
            MAJOR_REVISION_EXPECTED["synthetic_delta"],
            failures,
        )

    if "uncertainty" in required:
        checked += _verify_major_revision_table(
            "uncertainty",
            root / "results-uncertainty" / "tables" / "rank_biserial_uncertainty.csv",
            ("dataset", "predictor", "k"),
            MAJOR_REVISION_EXPECTED["uncertainty"],
            failures,
        )
        checked += _verify_major_revision_table(
            "tau_uncertainty",
            root / "results-uncertainty" / "tables" / "tau_uncertainty.csv",
            ("dataset", "predictor", "metric"),
            MAJOR_REVISION_EXPECTED["tau_uncertainty"],
            failures,
        )
        checked += _verify_major_revision_table(
            "loo",
            root / "results-uncertainty" / "tables" / "correlation_leave_one_out.csv",
            ("left_out",),
            MAJOR_REVISION_EXPECTED["loo"],
            failures,
        )

    if "sensitivity" in required:
        checked += _verify_major_revision_table(
            "sensitivity",
            root / "results-sensitivity" / "tables" / "sensitivity_summary.csv",
            ("split_scope", "dataset", "predictor", "normalization"),
            MAJOR_REVISION_EXPECTED["sensitivity"],
            failures,
        )

    if "swat_highpass" in required:
        checked += _verify_major_revision_table(
            "swat_highpass",
            root / "results-swat-filter" / "tables" / "swat_highpass_summary.csv",
            ("dataset", "predictor", "seed", "filter", "filter_window"),
            MAJOR_REVISION_EXPECTED["swat_highpass"],
            failures,
        )

    if "public_deep" in required:
        path = root / "results-public-deep-smd" / "strr" / "tables" / "public_deep_best_mean_std.csv"
        if not path.exists():
            path = root / "tables" / "public_deep_best_mean_std.csv"
        if not path.exists():
            failures.append(f"public_deep: missing table {path}")
        else:
            frame = pd.read_csv(path)
            checked += _verify_rows("public_deep", frame, ("predictor", "postprocess", "top_n", "k"), PUBLIC_DEEP_EXPECTED, failures)
    elif not (root / "results-public-deep-smd" / "strr" / "tables" / "budget_curve_summary.csv").exists():
        warnings.append("public_deep: not checked; public deep score export table is absent")

    return {"passed": not failures, "checked": checked, "failures": failures, "warnings": warnings}


def _verify_major_revision_table(
    group: str,
    path: Path,
    keys: tuple[str, ...],
    expected_rows: list[dict[str, object]],
    failures: list[str],
) -> int:
    if not path.exists():
        failures.append(f"{group}: missing table {path}")
        return 0
    return _verify_rows(group, pd.read_csv(path), keys, expected_rows, failures)


def _synthetic_delta_summary(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("positive", frame[frame["delta_mu"] > 0.1]),
        ("near_equal", frame[frame["delta_mu"].abs() <= 0.1]),
        ("negative", frame[frame["delta_mu"] < -0.1]),
    ]
    rows = []
    for regime, subset in groups:
        rows.append(
            {
                "regime": regime,
                "mean_forward_gain": float(subset["forward_gain"].mean()),
                "mean_raw_f1": float(subset["raw_f1_raw"].mean()),
                "mean_rank_biserial": float(subset["rank_biserial"].mean()),
                "n": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def _verify_rows(group: str, frame: pd.DataFrame, keys: tuple[str, ...], expected_rows: list[dict[str, object]], failures: list[str]) -> int:
    checked = 0
    for expected in expected_rows:
        mask = pd.Series(True, index=frame.index)
        for key in keys:
            if key not in expected:
                continue
            mask &= frame[key].astype(str) == str(expected[key])
        rows = frame[mask]
        if rows.empty:
            failures.append(f"{group}: missing row { {key: expected.get(key) for key in keys} }")
            continue
        actual = rows.iloc[0]
        for column, expected_value in expected.items():
            if column in keys:
                continue
            if column not in frame.columns:
                failures.append(f"{group}: missing column {column}")
                continue
            if not _matches(actual[column], expected_value):
                failures.append(
                    f"{group}: { {key: expected.get(key) for key in keys} } column {column} "
                    f"expected {expected_value}, got {actual[column]}"
                )
        checked += 1
    return checked


def _matches(actual: object, expected: object) -> bool:
    if expected is None:
        return pd.isna(actual)
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        try:
            return int(actual) == int(expected)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, (float, np.floating)):
        try:
            actual_f = float(actual)
        except (TypeError, ValueError):
            return False
        tol = 0.1 if abs(float(expected)) >= 5 else 0.005
        return abs(actual_f - float(expected)) <= max(ROUND_TOL, tol)
    return str(actual) == str(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reproduced result tables against the paper numbers.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--require-public-deep", action="store_true")
    parser.add_argument("--require-major-revision", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    required = set(EXPECTED.keys())
    if args.require_public_deep:
        required.add("public_deep")
    if args.require_major_revision:
        required.update({"synthetic", "uncertainty", "sensitivity", "swat_highpass"})
    if (args.results_root / "results-rrp-detector-pilot").exists():
        required.add("negative")
    report = verify_paper_results(args.results_root, required_groups=required)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
