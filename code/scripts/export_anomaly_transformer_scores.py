from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hbpc.public_deep_export import (
    aggregate_window_scores_to_points,
    ensure_same_length,
    make_sliding_windows,
    resolve_time_series_library_arrays,
    save_score_npz,
)


def _patch_cuda_for_cpu() -> None:
    if torch.cuda.is_available():
        return
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def _kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def _load_model(repo_root: Path, win_size: int, channels: int) -> torch.nn.Module:
    _patch_cuda_for_cpu()
    sys.path.insert(0, str(repo_root))
    module = importlib.import_module("model.AnomalyTransformer")
    return module.AnomalyTransformer(
        win_size=win_size,
        enc_in=channels,
        c_out=channels,
        d_model=64,
        n_heads=4,
        e_layers=1,
        d_ff=64,
        dropout=0.0,
    ).float()


def _train(
    model: torch.nn.Module,
    train_windows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tensor = torch.as_tensor(train_windows, dtype=torch.float32)
    for _ in range(epochs):
        order = torch.randperm(tensor.shape[0])
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[order[start : start + batch_size]].to(device)
            optimizer.zero_grad()
            output, series, prior, _ = model(batch)
            series_loss = 0.0
            prior_loss = 0.0
            for idx in range(len(prior)):
                norm_prior = prior[idx] / torch.unsqueeze(torch.sum(prior[idx], dim=-1), dim=-1).repeat(
                    1, 1, 1, batch.shape[1]
                )
                series_loss = series_loss + torch.mean(_kl_loss(series[idx], norm_prior.detach()))
                prior_loss = prior_loss + torch.mean(_kl_loss(norm_prior, series[idx].detach()))
            rec_loss = torch.mean((output - batch) ** 2)
            loss = rec_loss - 3.0 * series_loss / max(1, len(prior)) + rec_loss + 3.0 * prior_loss / max(1, len(prior))
            loss.backward()
            optimizer.step()


def _score(model: torch.nn.Module, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    rows: list[np.ndarray] = []
    temperature = 50.0
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            output, series, prior, _ = model(batch)
            rec = torch.mean((output - batch) ** 2, dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for idx in range(len(prior)):
                norm_prior = prior[idx] / torch.unsqueeze(torch.sum(prior[idx], dim=-1), dim=-1).repeat(
                    1, 1, 1, batch.shape[1]
                )
                series_loss = series_loss + _kl_loss(series[idx], norm_prior.detach()) * temperature
                prior_loss = prior_loss + _kl_loss(norm_prior, series[idx].detach()) * temperature
            metric = torch.softmax(-(series_loss + prior_loss), dim=-1)
            rows.append((metric * rec).detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def export_anomaly_transformer_scores(
    repo_root: Path,
    data_root: Path,
    output_root: Path,
    dataset: str = "SMD",
    seed: int = 0,
    win_size: int = 100,
    train_limit: int = 10000,
    test_limit: int = 50000,
    train_step: int = 20,
    test_step: int = 1,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = resolve_time_series_library_arrays(data_root, dataset)
    train = np.load(arrays.train)[:train_limit].astype(np.float32)
    test = np.load(arrays.test)[:test_limit].astype(np.float32)
    labels = np.load(arrays.test_label)[:test_limit].astype(np.int64)
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train).astype(np.float32)
    test = scaler.transform(test).astype(np.float32)
    train_windows, _ = make_sliding_windows(train, win_size=win_size, step=train_step)
    test_windows, starts = make_sliding_windows(test, win_size=win_size, step=test_step)
    model = _load_model(repo_root, win_size=win_size, channels=train.shape[1])
    _train(model, train_windows, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    window_scores = _score(model, test_windows, batch_size=batch_size, device=device)
    point_scores = aggregate_window_scores_to_points(window_scores, starts=starts, length=test.shape[0])
    point_scores, point_labels = ensure_same_length(point_scores, labels)
    out = output_root / dataset / "AnomalyTransformer" / str(seed) / "scores.npz"
    save_score_npz(out, scores=point_scores, labels=point_labels)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Anomaly Transformer public model scores.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--win-size", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=50000)
    parser.add_argument("--train-step", type=int, default=20)
    parser.add_argument("--test-step", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    export_anomaly_transformer_scores(
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_root=args.output_root,
        dataset=args.dataset,
        seed=args.seed,
        win_size=args.win_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        train_step=args.train_step,
        test_step=args.test_step,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
