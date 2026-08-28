from __future__ import annotations

import numpy as np
from pathlib import Path

from hbpc.cluster_inference import event_cluster_bootstrap_metrics, paired_bootstrap_ci, unique_artifacts_by_sha256


def test_paired_bootstrap_resamples_pairs_not_timestamps():
    result = paired_bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]), n_boot=500, seed=17)
    assert result["unit"] == "matched_event_pair"
    assert result["n"] == 4
    assert result["estimate"] == 2.5
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_event_cluster_bootstrap_keeps_whole_anomaly_events():
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    predictions = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    result = event_cluster_bootstrap_metrics(labels, predictions, n_boot=200, seed=17)
    assert result["unit"] == "anomaly_event_and_normal_block"
    assert result["anomaly_events"] == 2
    assert {"raw_f1", "event_recall", "mttd"} <= set(result["metrics"])


def test_duplicate_seed_artifacts_contribute_once(tmp_path: Path):
    first = tmp_path / "0.npz"
    second = tmp_path / "1.npz"
    first.write_bytes(b"same deterministic artifact")
    second.write_bytes(b"same deterministic artifact")
    unique, duplicates = unique_artifacts_by_sha256([first, second])
    assert unique == [first]
    assert duplicates == {second: first}
