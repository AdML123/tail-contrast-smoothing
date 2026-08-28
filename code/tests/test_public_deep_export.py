from pathlib import Path
import subprocess
import sys

import numpy as np

from hbpc.public_deep_export import (
    aggregate_window_scores_to_points,
    ensure_same_length,
    make_sliding_windows,
    resolve_time_series_library_arrays,
    save_score_npz,
)


def test_aggregate_window_scores_to_points_averages_overlaps():
    window_scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ]
    )

    point_scores = aggregate_window_scores_to_points(window_scores, starts=np.array([0, 1]), length=4)

    np.testing.assert_allclose(point_scores, np.array([1.0, 6.0, 11.5, 30.0]))


def test_make_sliding_windows_uses_requested_step_and_starts():
    data = np.arange(12, dtype=float).reshape(6, 2)

    windows, starts = make_sliding_windows(data, win_size=3, step=2)

    assert starts.tolist() == [0, 2]
    assert windows.shape == (2, 3, 2)
    np.testing.assert_allclose(windows[1], data[2:5])


def test_save_score_npz_writes_benchmark_compatible_arrays(tmp_path: Path):
    output = tmp_path / "scores.npz"

    save_score_npz(output, scores=np.array([0.1, 0.2]), labels=np.array([0, 1]))

    saved = np.load(output)
    assert saved["scores"].tolist() == [0.1, 0.2]
    assert saved["labels"].tolist() == [0, 1]


def test_ensure_same_length_trims_leading_unscored_prefix():
    scores = np.array([np.nan, np.nan, 0.3, 0.4])
    labels = np.array([0, 1, 0, 1])

    trimmed_scores, trimmed_labels = ensure_same_length(scores, labels)

    np.testing.assert_allclose(trimmed_scores, np.array([0.3, 0.4]))
    assert trimmed_labels.tolist() == [0, 1]


def test_resolve_time_series_library_arrays_accepts_huggingface_layout(tmp_path: Path):
    dataset_dir = tmp_path / "SMD"
    dataset_dir.mkdir()
    for suffix in ["train", "test", "test_label"]:
        (dataset_dir / f"SMD_{suffix}.npy").write_bytes(b"fake")

    paths = resolve_time_series_library_arrays(tmp_path, "SMD")

    assert paths.train == dataset_dir / "SMD_train.npy"
    assert paths.test == dataset_dir / "SMD_test.npy"
    assert paths.test_label == dataset_dir / "SMD_test_label.npy"


def test_resolve_time_series_library_arrays_accepts_flat_layout(tmp_path: Path):
    for suffix in ["train", "test", "test_label"]:
        (tmp_path / f"SMD_{suffix}.npy").write_bytes(b"fake")

    paths = resolve_time_series_library_arrays(tmp_path, "SMD")

    assert paths.train == tmp_path / "SMD_train.npy"
    assert paths.test == tmp_path / "SMD_test.npy"
    assert paths.test_label == tmp_path / "SMD_test_label.npy"


def test_public_export_scripts_are_directly_executable_help():
    for script in [
        "scripts/export_tsl_public_scores.py",
        "scripts/export_anomaly_transformer_scores.py",
    ]:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
