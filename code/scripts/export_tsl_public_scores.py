from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _args(model: str, channels: int, win_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=win_size,
        pred_len=0,
        top_k=3,
        num_kernels=3,
        enc_in=channels,
        c_out=channels,
        d_model=32,
        embed="timeF",
        freq="h",
        dropout=0.1,
        e_layers=1,
        d_ff=32,
        n_heads=4,
        factor=1,
        activation="gelu",
        output_attention=False,
        moving_avg=25,
        distil=True,
        label_len=0,
        d_layers=1,
        expand=2,
        d_conv=4,
        model=model,
    )


def _load_model(tsl_root: Path, model_name: str, channels: int, win_size: int) -> torch.nn.Module:
    sys.path.insert(0, str(tsl_root))
    module = importlib.import_module(f"models.{model_name}")
    model = module.Model(_args(model_name, channels=channels, win_size=win_size)).float()
    return model


def _train_autoencoder(
    model: torch.nn.Module,
    train_windows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    tensor = torch.as_tensor(train_windows, dtype=torch.float32)
    for _ in range(epochs):
        order = torch.randperm(tensor.shape[0])
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[order[start : start + batch_size]].to(device)
            optim.zero_grad()
            output = model(batch, None, None, None)
            loss = torch.mean((output - batch) ** 2)
            loss.backward()
            optim.step()


def _score_windows(model: torch.nn.Module, windows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            output = model(batch, None, None, None)
            score = torch.mean((output - batch) ** 2, dim=-1)
            rows.append(score.detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def export_tsl_scores(
    tsl_root: Path,
    data_root: Path,
    output_root: Path,
    models: list[str],
    dataset: str = "SMD",
    seed: int = 0,
    win_size: int = 100,
    train_limit: int = 10000,
    test_limit: int = 50000,
    train_step: int = 20,
    test_step: int = 1,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
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
    for model_name in models:
        model = _load_model(tsl_root, model_name, channels=train.shape[1], win_size=win_size)
        _train_autoencoder(model, train_windows, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
        window_scores = _score_windows(model, test_windows, batch_size=batch_size, device=device)
        point_scores = aggregate_window_scores_to_points(window_scores, starts=starts, length=test.shape[0])
        point_scores, point_labels = ensure_same_length(point_scores, labels)
        out = output_root / dataset / model_name / str(seed) / "scores.npz"
        save_score_npz(out, scores=point_scores, labels=point_labels)
        print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Time-Series-Library public model scores.")
    parser.add_argument("--tsl-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["TimesNet", "Transformer", "Autoformer"])
    parser.add_argument("--dataset", default="SMD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--win-size", type=int, default=100)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=50000)
    parser.add_argument("--train-step", type=int, default=20)
    parser.add_argument("--test-step", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    export_tsl_scores(
        tsl_root=args.tsl_root,
        data_root=args.data_root,
        output_root=args.output_root,
        models=list(args.models),
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
