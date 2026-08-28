from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ("SMD", "MSL", "SMAP", "PSM", "SWaT")
DEFAULT_SEEDS = ("0", "1", "2")


def run(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper tables and figures from a cold checkout.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets") / "Time-Series-Library")
    parser.add_argument("--results-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-public-deep", action="store_true")
    parser.add_argument("--anomaly-transformer-root", type=Path, default=Path("external") / "Anomaly-Transformer")
    parser.add_argument("--tsl-root", type=Path, default=Path("external") / "Time-Series-Library")
    parser.add_argument("--public-deep-epochs", type=str, default="1")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root

    if not args.skip_download:
        run(
            [
                sys.executable,
                "scripts/download_time_series_library.py",
                "--local-dir",
                str(dataset_root),
            ],
            cwd=repo,
            dry_run=args.dry_run,
        )

    one_step_dir = args.results_root / "results-five-dataset-one-step"
    run(
        [
            sys.executable,
            "-m",
            "hbpc.run",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(one_step_dir),
            "--datasets",
            *DEFAULT_DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *DEFAULT_SEEDS,
            "--lookback",
            "100",
            "--epochs",
            "10",
            "--device",
            args.device,
            "--max-train-points",
            "10000",
            "--max-test-points",
            "50000",
        ],
        cwd=repo,
        dry_run=args.dry_run,
    )

    strr_dir = args.results_root / "results-strr-five-dataset"
    run(
        [
            sys.executable,
            "-m",
            "hbpc.supplement_experiments",
            "--score-roots",
            str(one_step_dir / "raw"),
            "--output-dir",
            str(strr_dir),
            "--datasets",
            *DEFAULT_DATASETS,
            "--benchmark-datasets",
            *DEFAULT_DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *DEFAULT_SEEDS,
        ],
        cwd=repo,
        dry_run=args.dry_run,
    )

    adaptation_dir = args.results_root / "results-adaptation-causal-score-artifact"
    run(
        [
            sys.executable,
            "-m",
            "hbpc.adaptation_causal_experiments",
            "--score-root",
            str(one_step_dir / "raw"),
            "--output-dir",
            str(adaptation_dir),
            "--datasets",
            *DEFAULT_DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *DEFAULT_SEEDS,
            "--horizons",
            "3",
            "5",
        ],
        cwd=repo,
        dry_run=args.dry_run,
    )

    generated_figures = args.results_root / "generated-paper-figures"
    run(
        [
            sys.executable,
            "scripts/generate_mechanism_figures.py",
            "--output-dir",
            str(generated_figures),
        ],
        cwd=repo,
        dry_run=args.dry_run,
    )
    run(
        [
            sys.executable,
            "scripts/generate_case_comparison.py",
            "--scores-path",
            str(one_step_dir / "raw" / "SMD" / "one_step" / "0" / "scores.npz"),
            "--output-dir",
            str(generated_figures),
            "--k",
            "3",
            "--top-n",
            "300",
        ],
        cwd=repo,
        dry_run=args.dry_run,
    )


    if args.include_public_deep:
        public_raw = args.results_root / "results-public-deep-smd" / "raw"
        for seed in DEFAULT_SEEDS:
            run(
                [
                    sys.executable,
                    "scripts/export_anomaly_transformer_scores.py",
                    "--repo-root",
                    str(args.anomaly_transformer_root),
                    "--data-root",
                    str(dataset_root),
                    "--output-root",
                    str(public_raw),
                    "--seed",
                    seed,
                    "--train-limit",
                    "10000",
                    "--test-limit",
                    "50000",
                    "--epochs",
                    args.public_deep_epochs,
                ],
                cwd=repo,
                dry_run=args.dry_run,
            )
            run(
                [
                    sys.executable,
                    "scripts/export_tsl_public_scores.py",
                    "--tsl-root",
                    str(args.tsl_root),
                    "--data-root",
                    str(dataset_root),
                    "--output-root",
                    str(public_raw),
                    "--models",
                    "TimesNet",
                    "Transformer",
                    "Autoformer",
                    "--seed",
                    seed,
                    "--train-limit",
                    "10000",
                    "--test-limit",
                    "50000",
                    "--epochs",
                    args.public_deep_epochs,
                ],
                cwd=repo,
                dry_run=args.dry_run,
            )

        run(
            [
                sys.executable,
                "-m",
                "hbpc.supplement_experiments",
                "--score-roots",
                str(public_raw),
                "--output-dir",
                str(args.results_root / "results-public-deep-smd" / "strr"),
                "--datasets",
                "SMD",
                "--benchmark-datasets",
                "SMD",
                "--methods",
                "TimesNet",
                "Transformer",
                "Autoformer",
                "AnomalyTransformer",
                "--seeds",
                *DEFAULT_SEEDS,
            ],
            cwd=repo,
            dry_run=args.dry_run,
        )

    print("\nReproduction stages completed.")


if __name__ == "__main__":
    main()
