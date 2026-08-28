from __future__ import annotations

import numpy as np

from hbpc.synthetic_regime_experiments import (
    SyntheticSetting,
    generate_synthetic_score_series,
    run_synthetic_regime_grid,
)


def test_synthetic_generator_uses_matched_peak_distributions():
    setting = SyntheticSetting(mu_normal=0.2, mu_anomaly=1.8, phi=0.2, noise_family="gaussian", seed=11)

    series = generate_synthetic_score_series(setting)

    normal_peaks = series.scores[series.normal_event_starts]
    anomaly_peaks = series.scores[series.anomaly_event_starts]
    assert series.scores.shape == series.labels.shape
    assert len(normal_peaks) == len(anomaly_peaks)
    assert abs(float(np.mean(normal_peaks) - np.mean(anomaly_peaks))) < 0.25


def test_synthetic_forward_gain_tracks_tail_location_difference(tmp_path):
    table, heatmap = run_synthetic_regime_grid(
        output_dir=tmp_path,
        mu_normals=(0.2, 1.4),
        mu_anomalies=(0.2, 1.4),
        phis=(0.2,),
        noise_families=("gaussian",),
        seeds=(0, 1),
        length=3500,
        events_per_class=16,
        top_n=24,
        k=3,
    )

    positive = table[table["delta_mu"] > 0.5]["forward_gain"].mean()
    negative = table[table["delta_mu"] < -0.5]["forward_gain"].mean()

    assert positive > negative
    assert {"mu_normal", "mu_anomaly", "delta_mu", "forward_gain", "rank_biserial"}.issubset(table.columns)
    assert heatmap.shape[0] == 4
    assert (tmp_path / "tables" / "synthetic_regime_grid.csv").is_file()
    assert (tmp_path / "figures" / "synthetic_delta_gain_heatmap.png").is_file()


def test_synthetic_grid_reports_independent_replicates(tmp_path):
    rows, summary = run_synthetic_regime_grid(
        output_dir=tmp_path,
        mu_normals=(0.2,), mu_anomalies=(1.4,), phis=(0.2,),
        noise_families=("gaussian",), seeds=tuple(range(8)),
        length=3500, events_per_class=16, top_n=24, k=3,
    )
    assert rows["replicate"].nunique() == 8
    assert summary.iloc[0]["n_replicates"] == 8
    assert {"gain_ci_low", "gain_ci_high", "predicted_sign_fraction"} <= set(summary.columns)


def test_each_replicate_uses_identical_peak_draws_between_classes():
    setting = SyntheticSetting(mu_normal=0.2, mu_anomaly=1.4, phi=0.2, noise_family="gaussian", seed=11)
    series = generate_synthetic_score_series(setting)
    normal = series.scores[series.normal_event_starts]
    anomaly = series.scores[series.anomaly_event_starts]
    assert np.array_equal(normal, anomaly)
