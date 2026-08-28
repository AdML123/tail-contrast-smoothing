from __future__ import annotations

import inspect
import numpy as np
import pytest
from pathlib import Path

from hbpc.score_benchmark import ScoreRun
from hbpc.data import TimeSeriesDataset
from hbpc.experiments import _one_step_scores_segmented
from hbpc.score_benchmark import backward_average_score
from hbpc.spl_experiments import _segmentwise_transform
from hbpc.spl_experiments import PRIMARY_PROTOCOL, analyze_dataset, fixed_alarm_count, load_unique_score_runs, run_corrected_analysis, selected_alarm_indices


def test_default_corrected_analysis_covers_the_preregistered_six_datasets():
    default = inspect.signature(run_corrected_analysis).parameters["datasets"].default
    assert tuple(default) == ("SMD", "MSL", "SMAP", "PSM", "SWaT", "HAI")


def test_primary_alarm_budget_is_half_percent_without_label_input():
    assert fixed_alarm_count(length=50_000, fraction=0.005) == 250
    assert fixed_alarm_count(length=101, fraction=0.005) == 1


def test_primary_protocol_is_not_a_test_selected_grid():
    assert PRIMARY_PROTOCOL.k == 3
    assert PRIMARY_PROTOCOL.alarm_fraction == 0.005
    assert PRIMARY_PROTOCOL.ewma_alpha == 0.3
    assert PRIMARY_PROTOCOL.peak_caliper == 0.2


def test_dataset_analysis_reports_balance_contrast_and_delay_metrics():
    scores = np.zeros(50, dtype=float)
    scores[[2, 10, 20, 28, 38]] = [5.1, 5.11, 5.2, 5.19, 5.0]
    scores[[11, 12, 29, 30]] = [3.0, 2.0, 3.2, 2.1]
    labels = np.zeros(50, dtype=bool)
    labels[10:13] = True
    labels[28:31] = True
    run = ScoreRun(dataset="toy", predictor="p", seed=0, scores=scores, labels=labels)
    result = analyze_dataset(run, training_normal_scores=np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.8]), protocol=PRIMARY_PROTOCOL)
    assert {"dataset", "matched_pairs", "peak_smd", "tail_contrast", "contrast_ci_low", "contrast_ci_high", "method", "raw_f1", "raw_f1_ci_low", "raw_f1_ci_high", "event_recall", "event_recall_ci_low", "event_recall_ci_high", "mttd", "mttd_ci_low", "mttd_ci_high", "regime", "evidence_status"} <= set(result.columns)
    assert set(result["method"]) == {"raw_realtime", "raw_delayed", "confirmation_mean", "ewma"}


def test_selected_alarm_indices_depend_only_on_scores():
    scores = np.linspace(0.0, 1.0, 1000)
    first = selected_alarm_indices(scores, fraction=0.005)
    second = selected_alarm_indices(scores.copy(), fraction=0.005)
    assert np.array_equal(first, second)


def test_five_dataset_loader_requires_training_normal_scores(tmp_path: Path):
    path = tmp_path / "SMD" / "one_step" / "0"
    path.mkdir(parents=True)
    np.savez(path / "scores.npz", scores=np.zeros(4), labels=np.zeros(4))
    with pytest.raises(ValueError, match="training_normal_scores"):
        load_unique_score_runs([path / "scores.npz"])


def test_five_dataset_loader_deduplicates_identical_seed_artifacts(tmp_path: Path):
    paths = []
    for seed in range(3):
        path = tmp_path / "SMD" / "one_step" / str(seed)
        path.mkdir(parents=True)
        np.savez(path / "scores.npz", scores=np.linspace(0.0, 1.0, 40), labels=np.zeros(40, dtype=bool), training_normal_scores=np.linspace(0.0, 0.5, 20))
        paths.append(path / "scores.npz")
    runs, audit = load_unique_score_runs(paths)
    assert len(runs) == 1
    assert audit["input_artifacts"] == 3
    assert audit["unique_artifacts"] == 1


def test_one_step_segmented_scoring_resets_warmup_at_each_run() -> None:
    train = np.arange(40, dtype=float).reshape(20, 2)
    segments = (
        np.arange(20, dtype=float).reshape(10, 2),
        np.arange(20, 40, dtype=float).reshape(10, 2),
    )
    scores, training_scores, params = _one_step_scores_segmented(
        train=train,
        test_segments=segments,
        lookback=3,
        epochs=1,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
    )
    assert len(scores) == 20
    assert np.isnan(scores[0:3]).all()
    assert np.isnan(scores[10:13]).all()
    assert np.isfinite(scores[3:10]).all()
    assert np.isfinite(scores[13:]).all()
    assert len(training_scores) == len(train)
    assert params == 8


def test_segmentwise_transform_does_not_read_previous_run() -> None:
    scores = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    transformed = _segmentwise_transform(scores, (3, 3), lambda values: backward_average_score(values, k=2))
    assert np.isnan(transformed[0])
    assert transformed[1] == 1.5
    assert np.isnan(transformed[3])
    assert transformed[4] == 150.0
