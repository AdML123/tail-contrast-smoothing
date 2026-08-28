from __future__ import annotations

from pathlib import Path
import gzip

import numpy as np
import pandas as pd
import pytest

from hbpc.data import _read_csv_values, load_dataset, missing_files


def test_read_csv_values_treats_empty_numeric_cells_as_nan(tmp_path: Path) -> None:
    csv_path = tmp_path / "values.csv"
    csv_path.write_text(
        "timestamp,feature_0,feature_1\n"
        "0.0,1.5,\n"
        "1.0,,2.5\n",
        encoding="utf-8",
    )

    values = _read_csv_values(csv_path)

    assert values.shape == (2, 2)
    assert values[0, 0] == 1.5
    assert values[1, 1] == 2.5
    assert np.isnan(values[0, 1])
    assert np.isnan(values[1, 0])


def _write_hai_file(path: Path, start: str, rows: int, attacked: tuple[int, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(start=start, periods=rows, freq="s")
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "P1_A": np.arange(rows, dtype=float),
            "P1_B": np.arange(rows, dtype=float) + 10,
            "P1_C": np.arange(rows, dtype=float) + 20,
            "attack": [1 if index in attacked else 0 for index in range(rows)],
            "attack_P1": 0,
            "attack_P2": 0,
            "attack_P3": 0,
        }
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def test_hai_loader_preserves_test_segment_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "hai-master"
    data_dir = root / "hai-21.03"
    for index in range(1, 4):
        _write_hai_file(data_dir / f"train{index}.csv.gz", f"2020-01-0{index} 00:00:00", 8)
    for index in range(1, 6):
        _write_hai_file(data_dir / f"test{index}.csv.gz", f"2020-02-0{index} 00:00:00", 8, attacked=(2, 3))

    dataset = load_dataset(root, "HAI")

    assert dataset.name == "HAI"
    assert dataset.train.shape == (24, 3)
    assert dataset.test_segments is not None
    assert len(dataset.test_segments) == 5
    assert [segment.shape for segment in dataset.test_segments] == [(8, 3)] * 5
    assert dataset.labels.sum() == 10
    assert [labels.sum() for labels in dataset.label_segments] == [2] * 5
    assert missing_files(root, "HAI") == ()


def test_hai_loader_rejects_irregular_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "hai-master"
    data_dir = root / "hai-21.03"
    for index in range(1, 4):
        _write_hai_file(data_dir / f"train{index}.csv.gz", f"2020-01-0{index} 00:00:00", 8)
    for index in range(1, 6):
        _write_hai_file(data_dir / f"test{index}.csv.gz", f"2020-02-0{index} 00:00:00", 8)
    broken = data_dir / "test3.csv.gz"
    frame = pd.read_csv(broken, compression="gzip")
    frame.loc[4, "time"] = frame.loc[3, "time"]
    with gzip.open(broken, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)

    with pytest.raises(ValueError, match="invalid or duplicate timestamps"):
        load_dataset(root, "HAI")


def test_hai_loader_rejects_missing_attack_indicator(tmp_path: Path) -> None:
    root = tmp_path / "hai-master"
    data_dir = root / "hai-21.03"
    for index in range(1, 4):
        _write_hai_file(data_dir / f"train{index}.csv.gz", f"2020-01-0{index} 00:00:00", 8)
    for index in range(1, 6):
        path = data_dir / f"test{index}.csv.gz"
        _write_hai_file(path, f"2020-02-0{index} 00:00:00", 8)
        frame = pd.read_csv(path, compression="gzip").drop(columns=["attack"])
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)

    with pytest.raises(ValueError, match="attack indicator"):
        load_dataset(root, "HAI")
