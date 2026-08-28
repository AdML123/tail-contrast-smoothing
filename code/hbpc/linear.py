import numpy as np
import torch
from torch import nn


def parameter_count(channels: int, lookback: int, horizons: int) -> int:
    return int(channels * horizons * (lookback + 1))


class _ChannelLinear(nn.Module):
    def __init__(self, channels: int, lookback: int, horizons: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, horizons, lookback))
        self.bias = nn.Parameter(torch.zeros(channels, horizons))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nlc,ckl->nkc", x, self.weight) + self.bias.transpose(0, 1).unsqueeze(0)


class LinearPredictor:
    def __init__(self, module: _ChannelLinear, device: str):
        self.module = module
        self.device = device

    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.module.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                outputs.append(self.module(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def train_linear_predictor(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> LinearPredictor:
    torch.manual_seed(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _n, lookback, channels = x.shape
    _n_y, horizons, y_channels = y.shape
    if y_channels != channels:
        raise ValueError("x and y must have the same channel count")

    module = _ChannelLinear(channels, lookback, horizons).to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((module(x_tensor) - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
    return LinearPredictor(module=module, device=device)
