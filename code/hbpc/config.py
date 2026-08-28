from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_root: Path = Path("datasets/Time-Series-Library")
    datasets: tuple[str, ...] = ("SMD", "MSL", "SMAP", "PSM", "SWaT")
    methods: tuple[str, ...] = (
        "one_step",
        "multi_mean_raw",
        "multi_mean_norm_rms_clip",
        "multi_mean_norm_median_iqr",
        "hbpc_full_rms_clip",
        "hbpc_full_median_iqr",
        "multi_mean_norm",
        "hbpc_full",
        "hbpc_magnitude",
        "hbpc_shape",
        "moving_average",
        "ar1",
        "var1",
    )
    lookback: int = 100
    horizons: int = 8
    eta: float = 1.0
    seeds: tuple[int, ...] = (0, 1, 2)
    train_epochs: int = 10
    learning_rate: float = 1e-3
    spot_q: float = 1e-3
    calibration_fraction: float = 0.10
    device: str = "cpu"