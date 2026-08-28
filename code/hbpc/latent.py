import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _dct_basis(patch_length: int) -> torch.Tensor:
    basis = torch.empty(patch_length, patch_length, dtype=torch.float32)
    factor = math.pi / patch_length
    for k in range(patch_length):
        scale = math.sqrt(1.0 / patch_length) if k == 0 else math.sqrt(2.0 / patch_length)
        for n in range(patch_length):
            basis[k, n] = scale * math.cos(factor * (n + 0.5) * k)
    return basis


class FixedDCTTargetEncoder(nn.Module):
    def __init__(self, patch_length: int, channels: int, latent_dim: int, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(patch_length * channels, latent_dim, generator=generator)
        projection = projection / torch.linalg.norm(projection, dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", _dct_basis(patch_length), persistent=False)
        self.register_buffer("projection", projection, persistent=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        coeffs = torch.einsum("kp,npc->nkc", self.basis.to(patches.device), patches)
        flat = coeffs.flatten(start_dim=1)
        z = flat @ self.projection.to(patches.device)
        return torch.nn.functional.normalize(z, dim=1)


def robust_normalize_scores(scores: np.ndarray, train_scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    valid = train_scores[np.isfinite(train_scores)]
    median = float(np.median(valid)) if valid.size else 0.0
    q25, q75 = np.percentile(valid, [25.0, 75.0]) if valid.size else (0.0, 1.0)
    iqr = float(q75 - q25)
    scale = iqr if iqr >= eps else 1.0
    return np.maximum((scores - median) / scale, 0.0)


@dataclass(frozen=True)
class LatentProfileConfig:
    lookback: int = 100
    patch_length: int = 8
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    latent_dim: int = 32
    hidden_dim: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    batch_size: int = 512
    target_seed: int = 0


class LatentPredictor(nn.Module):
    def __init__(self, channels: int, horizons: int, latent_dim: int, hidden_dim: int, target_seed: int = 0):
        super().__init__()
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.mix = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        self.head = nn.Linear(hidden_dim, horizons * latent_dim)
        self.horizons = horizons
        self.latent_dim = latent_dim
        self.target_seed = target_seed

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = context.transpose(1, 2)
        x = torch.relu(self.temporal(x))
        x = torch.relu(self.mix(x))
        x = x.mean(dim=2)
        out = self.head(x).reshape(len(context), self.horizons, self.latent_dim)
        return torch.nn.functional.normalize(out, dim=2)


def count_latent_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _make_latent_training_arrays(series: np.ndarray, cfg: LatentProfileConfig) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=np.float64)
    max_horizon = max(cfg.horizons)
    count = len(series) - cfg.lookback - max_horizon - cfg.patch_length + 2
    if count <= 0:
        raise ValueError("series is shorter than lookback + max horizon + patch length")
    x = np.empty((count, cfg.lookback, series.shape[1]), dtype=np.float32)
    patches = np.empty((count, len(cfg.horizons), cfg.patch_length, series.shape[1]), dtype=np.float32)
    for i in range(count):
        end = i + cfg.lookback
        x[i] = series[i:end]
        for h_i, horizon in enumerate(cfg.horizons):
            start = end + horizon - 1
            patches[i, h_i] = series[start : start + cfg.patch_length]
    return x, patches


def _target_latents(patches: np.ndarray, encoder: FixedDCTTargetEncoder, device: str) -> torch.Tensor:
    count, horizons, patch_length, channels = patches.shape
    flat = torch.as_tensor(
        patches.reshape(count * horizons, patch_length, channels),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        encoded = encoder(flat).reshape(count, horizons, -1)
    return encoded


def train_latent_predictor(
    train: np.ndarray,
    cfg: LatentProfileConfig,
    seed: int,
    device: str,
) -> LatentPredictor:
    torch.manual_seed(seed)
    train_x, train_patches = _make_latent_training_arrays(train, cfg)
    channels = train_x.shape[2]
    model = LatentPredictor(
        channels=channels,
        horizons=len(cfg.horizons),
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        target_seed=cfg.target_seed,
    ).to(device)
    target_encoder = FixedDCTTargetEncoder(
        patch_length=cfg.patch_length,
        channels=channels,
        latent_dim=cfg.latent_dim,
        seed=cfg.target_seed,
    ).to(device)
    target_latents = _target_latents(train_patches, target_encoder, device)
    contexts = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    batch_size = max(1, cfg.batch_size)
    for _ in range(cfg.epochs):
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            target = target_latents[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch) - target) ** 2)
            loss.backward()
            optimizer.step()
    return model


def _raw_latent_scores(
    model: LatentPredictor,
    series: np.ndarray,
    cfg: LatentProfileConfig,
    device: str,
) -> np.ndarray:
    contexts, patches = _make_latent_training_arrays(series, cfg)
    target_encoder = FixedDCTTargetEncoder(
        patch_length=cfg.patch_length,
        channels=contexts.shape[2],
        latent_dim=cfg.latent_dim,
        seed=cfg.target_seed,
    ).to(device)
    target_latents = _target_latents(patches, target_encoder, device)
    context_tensor = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context_tensor), cfg.batch_size):
            predictions.append(model(context_tensor[start : start + cfg.batch_size]).cpu())
    pred = torch.cat(predictions, dim=0)
    errors = torch.mean((pred - target_latents.cpu()) ** 2, dim=2).numpy()

    sums = np.zeros(len(series), dtype=np.float64)
    counts = np.zeros(len(series), dtype=np.float64)
    for i in range(errors.shape[0]):
        context_end = i + cfg.lookback
        for h_i, horizon in enumerate(cfg.horizons):
            target_t = context_end + horizon + cfg.patch_length - 2
            if target_t < len(series):
                sums[target_t] += errors[i, h_i]
                counts[target_t] += 1.0
    scores = np.full(len(series), np.nan, dtype=np.float64)
    valid = counts > 0
    scores[valid] = sums[valid] / counts[valid]
    scores[: cfg.lookback + max(cfg.horizons) + cfg.patch_length - 2] = np.nan
    return scores


def latent_profile_scores(
    model: LatentPredictor,
    train: np.ndarray,
    test: np.ndarray,
    cfg: LatentProfileConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_raw = _raw_latent_scores(model, train, cfg, device)
    test_raw = _raw_latent_scores(model, test, cfg, device)
    return robust_normalize_scores(test_raw, train_raw), test_raw
