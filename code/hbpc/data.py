import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILES: dict[str, tuple[Path, ...]] = {
    "SMD": (
        Path("SMD/SMD_train.npy"),
        Path("SMD/SMD_test.npy"),
        Path("SMD/SMD_test_label.npy"),
    ),
    "MSL": (
        Path("MSL/MSL_train.npy"),
        Path("MSL/MSL_test.npy"),
        Path("MSL/MSL_test_label.npy"),
    ),
    "SMAP": (
        Path("SMAP/SMAP_train.npy"),
        Path("SMAP/SMAP_test.npy"),
        Path("SMAP/SMAP_test_label.npy"),
    ),
    "PSM": (
        Path("PSM/train.csv"),
        Path("PSM/test.csv"),
        Path("PSM/test_label.csv"),
    ),
    "SWaT": (
        Path("SWaT/swat_train2.csv"),
        Path("SWaT/swat2.csv"),
    ),
    "HAI": tuple(
        Path("hai-21.03") / f"{split}{index}.csv.gz"
        for split, indices in (("train", range(1, 4)), ("test", range(1, 6)))
        for index in indices
    ),
}


def expected_files(dataset: str | None = None) -> tuple[Path, ...]:
    if dataset is not None:
        return DATASET_FILES[dataset]
    files: list[Path] = []
    for dataset_files in DATASET_FILES.values():
        files.extend(dataset_files)
    return tuple(files)


def missing_files(root: Path, dataset: str | None = None) -> tuple[Path, ...]:
    root = Path(root)
    return tuple(path for path in expected_files(dataset) if not (root / path).exists())


@dataclass(frozen=True)
class TimeSeriesDataset:
    name: str
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    test_segments: tuple[np.ndarray, ...] | None = None
    label_segments: tuple[np.ndarray, ...] | None = None

    @property
    def channel_count(self) -> int:
        return int(self.train.shape[1])


def _flatten_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return (labels > 0).astype(np.int64)
    return (labels > 0).any(axis=1).astype(np.int64)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.nan_to_num(np.asarray(train, dtype=np.float64))
    test = np.nan_to_num(np.asarray(test, dtype=np.float64))
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (test - mean) / std, mean, std


def _load_npy_triplet(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(root / dataset / f"{dataset}_train.npy")
    test = np.load(root / dataset / f"{dataset}_test.npy")
    labels = np.load(root / dataset / f"{dataset}_test_label.npy")
    return train, test, labels


def _read_csv_values(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(
        [[float(value) if value.strip() else np.nan for value in row[1:]] for row in rows[1:]],
        dtype=np.float64,
    )


def _load_psm(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = _read_csv_values(root / "PSM" / "train.csv")
    test = _read_csv_values(root / "PSM" / "test.csv")
    labels = _read_csv_values(root / "PSM" / "test_label.csv")
    return train, test, labels


def _load_swat(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_all = _read_csv_values(root / "SWaT" / "swat_train2.csv")
    test_all = _read_csv_values(root / "SWaT" / "swat2.csv")
    if train_all.shape[1] < 2 or test_all.shape[1] < 2:
        raise ValueError("SWaT CSV files must contain feature columns and a label column")
    return train_all[:, :-1], test_all[:, :-1], test_all[:, -1:]


def _read_hai_file(
    path: Path,
    expected_process_columns: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read one HAI 21.03 run and validate its local time and label contract."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, compression="infer")
    if "time" not in frame.columns or "attack" not in frame.columns:
        raise ValueError(f"{path} must contain time and attack indicator columns")
    timestamps = pd.to_datetime(frame["time"], errors="coerce")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError(f"{path} contains invalid or duplicate timestamps")
    deltas = timestamps.diff().dropna().dt.total_seconds().to_numpy()
    if len(deltas) and not np.allclose(deltas, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path} must have one-second timestamp spacing")
    label_columns = {column for column in frame.columns if column == "attack" or column.startswith("attack_")}
    process_columns = tuple(column for column in frame.columns if column not in label_columns and column != "time")
    if expected_process_columns is not None and process_columns != expected_process_columns:
        raise ValueError(f"{path} process-variable columns do not match the HAI training contract")
    if len(process_columns) < 3:
        raise ValueError(f"{path} must contain at least three process variables")
    values = frame.loc[:, process_columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().all(axis=0).any():
        raise ValueError(f"{path} contains a process variable with no numeric values")
    attack = pd.to_numeric(frame["attack"], errors="coerce")
    if attack.isna().any():
        raise ValueError(f"{path} attack indicator contains non-numeric values")
    return values.to_numpy(dtype=np.float64), (attack.to_numpy() > 0).astype(np.int64), process_columns


def _load_hai(root: Path) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    root = Path(root) / "hai-21.03"
    train_arrays: list[np.ndarray] = []
    process_columns: tuple[str, ...] | None = None
    for index in range(1, 4):
        values, labels, process_columns = _read_hai_file(root / f"train{index}.csv.gz", process_columns)
        if labels.any():
            raise ValueError(f"HAI train{index} contains attack labels")
        train_arrays.append(values)
    test_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    for index in range(1, 6):
        values, labels, _ = _read_hai_file(root / f"test{index}.csv.gz", process_columns)
        test_arrays.append(values)
        label_arrays.append(labels)
    return np.concatenate(train_arrays, axis=0), tuple(test_arrays), tuple(label_arrays)


def load_dataset(root: Path, dataset: str) -> TimeSeriesDataset:
    root = Path(root)
    if dataset == "HAI":
        train_raw, test_segments_raw, labels_segments = _load_hai(root)
        test_raw = np.concatenate(test_segments_raw, axis=0)
        train, test, mean, std = _standardize(train_raw, test_raw)
        lengths = np.cumsum([len(segment) for segment in test_segments_raw[:-1]])
        test_segments = tuple(np.split(test, lengths)) if len(lengths) else (test,)
        labels = np.concatenate(labels_segments, axis=0)
        return TimeSeriesDataset(dataset, train, test, labels, mean, std, test_segments, labels_segments)
    if dataset in {"SMD", "MSL", "SMAP"}:
        train_raw, test_raw, labels_raw = _load_npy_triplet(root, dataset)
    elif dataset == "PSM":
        train_raw, test_raw, labels_raw = _load_psm(root)
    elif dataset == "SWaT":
        train_raw, test_raw, labels_raw = _load_swat(root)
    else:
        raise KeyError(f"Unsupported dataset: {dataset}")

    train, test, mean, std = _standardize(train_raw, test_raw)
    labels = _flatten_labels(labels_raw)
    if len(labels) != len(test):
        raise ValueError(f"{dataset} label length {len(labels)} does not match test length {len(test)}")
    return TimeSeriesDataset(dataset, train, test, labels, mean, std)
