import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from hbpc.data import load_dataset, missing_files
from hbpc.experiments import SMD_GATE_METHODS, run_one_method, summarize_metrics, write_metrics, write_smd_ablation_gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HBPC lightweight experiments")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/Time-Series-Library"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", default=["SMD", "MSL", "SMAP", "PSM", "SWaT"])
    parser.add_argument("--methods", nargs="+", default=list(SMD_GATE_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--horizons", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--max-train-points", type=int, default=None)
    parser.add_argument("--max-test-points", type=int, default=None)
    parser.add_argument("--sweep-k", nargs="*", type=int, default=None)
    parser.add_argument("--sweep-lookback", nargs="*", type=int, default=None)
    parser.add_argument("--sweep-eta", nargs="*", type=float, default=None)
    parser.add_argument("--sweep-calibration", nargs="*", type=float, default=None)
    return parser


def _truncate_dataset(dataset, max_train_points: int | None, max_test_points: int | None):
    train = dataset.train
    test = dataset.test
    labels = dataset.labels
    if max_train_points is not None:
        train = train[:max_train_points]
    if max_test_points is not None:
        test = test[:max_test_points]
        labels = labels[:max_test_points]
    return replace(dataset, train=train, test=test, labels=labels)


def main() -> None:
    args = _parser().parse_args()
    rows = []
    for dataset_name in args.datasets:
        missing = missing_files(args.dataset_root, dataset_name)
        if missing:
            message = f"{dataset_name} missing files: {', '.join(str(path) for path in missing)}"
            if args.skip_missing:
                print(f"Skipping {message}")
                continue
            raise FileNotFoundError(message)
        dataset = load_dataset(args.dataset_root, dataset_name)
        dataset = _truncate_dataset(dataset, args.max_train_points, args.max_test_points)
        for method in args.methods:
            for seed in args.seeds:
                rows.append(
                    run_one_method(
                        dataset=dataset,
                        method=method,
                        output_dir=args.output_dir,
                        lookback=args.lookback,
                        horizons=args.horizons,
                        eta=args.eta,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        seed=seed,
                        device=args.device,
                        calibration_fraction=args.calibration_fraction,
                    )
                )
    write_metrics(rows, args.output_dir / "metrics" / "lightweight_metrics.csv")
    if rows:
        frame = pd.DataFrame(rows)
        summarize_metrics(frame, args.output_dir / "tables" / "table_ii_lightweight.csv")
        summarize_metrics(frame, args.output_dir / "tables" / "table_iii_ablation.csv")
        write_smd_ablation_gate(frame, args.output_dir / "tables" / "smd_ablation_gate.csv")


if __name__ == "__main__":
    main()
