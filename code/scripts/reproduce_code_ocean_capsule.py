from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hbpc.paper_result_verifier import verify_paper_results


DATASETS = ("SMD", "MSL", "SMAP", "PSM", "SWaT")
SEEDS = ("0", "1", "2")
PUBLIC_DEEP_METHODS = ("TimesNet", "Transformer", "Autoformer", "AnomalyTransformer")


def expected_outputs() -> list[str]:
    figures = [
        "mechanism_relaxation_toy.png",
        "mechanism_forward_average.png",
        "case_comparison_smd_one_step.png",
        "cross_dataset_rank_biserial.png",
        "tau_norm_vs_r.png",
        "relax_ecdf_K3.png",
        "relax_ecdf_K5.png",
        "relax_ecdf_K10.png",
        "relax_ecdf_K20.png",
        "relaxation_curve_K3.png",
        "relaxation_curve_K5.png",
        "relaxation_curve_K10.png",
        "relaxation_curve_K20.png",
        "budget_curve_smd_one_step.png",
        "window_sensitivity_smd.png",
        "budget_curve_smd_AnomalyTransformer.png",
        "budget_curve_smd_Autoformer.png",
        "budget_curve_smd_TimesNet.png",
        "budget_curve_smd_Transformer.png",
        "synthetic_delta_gain_heatmap.png",
        "rank_biserial_ci_forest.png",
        "sensitivity_k_topn_curves.png",
        "swat_highpass_regime_movement.png",
    ]
    tables = [
        "notation.csv",
        "scope.csv",
        "regime_summary.csv",
        "cross_dataset_phenomenon.csv",
        "budget_curve_summary.csv",
        "delay_fairness.csv",
        "adaptation_dataset_summary.csv",
        "adaptation_dataset_correlation_summary.csv",
        "smd_score_sources.csv",
        "public_deep_budget_curve_summary.csv",
        "public_deep_best_mean_std.csv",
        "negative_variants.csv",
        "synthetic_regime_summary.csv",
        "synthetic_delta_gain_heatmap.csv",
        "synthetic_delta_summary.csv",
        "rank_biserial_uncertainty.csv",
        "tau_uncertainty.csv",
        "correlation_leave_one_out.csv",
        "sensitivity_summary.csv",
        "swat_highpass_summary.csv",
    ]
    metrics = [
        "capsule_numeric_report.json",
        "paper_result_verification.json",
    ]
    return [
        *[f"figures/{name}" for name in figures],
        *[f"tables/{name}" for name in tables],
        *[f"metrics/{name}" for name in metrics],
        "capsule_manifest.csv",
        "capsule_verification.json",
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce short-tail location paper artifacts inside a Code Ocean capsule.")
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--results-dir", type=Path, default=Path("../results"))
    parser.add_argument("--strict", action="store_true", help="Fail if any expected artifact or paper number is missing.")
    parser.add_argument("--skip-heavy", action="store_true", help="Accepted for compatibility; this capsule uses score artifacts only.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    started = time.perf_counter()

    _prepare_results(results_dir)
    artifacts = data_dir / "score-artifacts"
    one_step_scores = _require_dir(artifacts / "results-five-dataset-one-step" / "raw")
    smd_gate_scores = _require_dir(artifacts / "results-smd-gate" / "raw")
    public_deep_scores = _require_dir(artifacts / "results-public-deep-smd" / "raw")
    full_one_step_scores = artifacts / "results-five-dataset-one-step-full" / "raw"

    run_report: list[dict[str, object]] = []

    _run(
        [
            sys.executable,
            "-m",
            "hbpc.supplement_experiments",
            "--score-roots",
            str(one_step_scores),
            "--output-dir",
            str(results_dir / "results-strr-five-dataset"),
            "--datasets",
            *DATASETS,
            "--benchmark-datasets",
            *DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *SEEDS,
        ],
        repo,
        run_report,
        "five_dataset_strr",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.adaptation_causal_experiments",
            "--score-root",
            str(one_step_scores),
            "--output-dir",
            str(results_dir / "results-adaptation-causal-score-artifact"),
            "--datasets",
            *DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *SEEDS,
            "--horizons",
            "3",
            "5",
            "10",
            "20",
        ],
        repo,
        run_report,
        "adaptation_causal",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.rrp_experiments",
            "--scores-root",
            str(smd_gate_scores),
            "--output-dir",
            str(results_dir / "results-rrp-smd-phenomenon"),
            "--dataset",
            "SMD",
            "--detector",
            "one_step",
            "--seeds",
            *SEEDS,
        ],
        repo,
        run_report,
        "rrp_phenomenon",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.alignment_experiments",
            "--scores-root",
            str(smd_gate_scores),
            "--output-dir",
            str(results_dir / "results-strr-alignment"),
            "--dataset",
            "SMD",
            "--methods",
            "one_step",
            "multi_mean_raw",
            "multi_mean_norm_rms_clip",
            "hbpc_full_rms_clip",
            "--seeds",
            *SEEDS,
        ],
        repo,
        run_report,
        "score_source_alignment",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.supplement_experiments",
            "--score-roots",
            str(public_deep_scores),
            "--output-dir",
            str(results_dir / "results-public-deep-smd" / "strr"),
            "--datasets",
            "SMD",
            "--benchmark-datasets",
            "SMD",
            "--methods",
            *PUBLIC_DEEP_METHODS,
            "--seeds",
            *SEEDS,
        ],
        repo,
        run_report,
        "public_deep_strr",
    )

    _run(
        [
            sys.executable,
            "-m",
            "hbpc.synthetic_regime_experiments",
            "--output-dir",
            str(results_dir / "results-synthetic-regime"),
            "--length",
            "12000",
            "--events-per-class",
            "40",
            "--top-n",
            "60",
            "--k",
            "3",
        ],
        repo,
        run_report,
        "synthetic_regime",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.uncertainty_experiments",
            "--score-root",
            str(one_step_scores),
            "--output-dir",
            str(results_dir / "results-uncertainty"),
            "--adaptation-rows-path",
            str(results_dir / "results-adaptation-causal-score-artifact" / "metrics" / "adaptation_correlation_rows.csv"),
            "--datasets",
            *DATASETS,
            "--methods",
            "one_step",
            "--seeds",
            *SEEDS,
            "--horizons",
            "3",
            "5",
            "10",
            "20",
            "--n-boot",
            "1000",
        ],
        repo,
        run_report,
        "uncertainty",
    )
    sensitivity_cmd = [
        sys.executable,
        "-m",
        "hbpc.sensitivity_experiments",
        "--score-root",
        str(one_step_scores),
        "--output-dir",
        str(results_dir / "results-sensitivity"),
        "--datasets",
        *DATASETS,
        "--methods",
        "one_step",
        "--seeds",
        *SEEDS,
    ]
    if full_one_step_scores.exists():
        sensitivity_cmd.extend(["--full-score-root", str(full_one_step_scores)])
    _run(
        sensitivity_cmd,
        repo,
        run_report,
        "sensitivity",
    )
    _run(
        [
            sys.executable,
            "-m",
            "hbpc.swat_filter_experiments",
            "--score-root",
            str(one_step_scores),
            "--output-dir",
            str(results_dir / "results-swat-filter"),
            "--seeds",
            *SEEDS,
        ],
        repo,
        run_report,
        "swat_highpass",
    )

    _run_negative_pilots(repo, results_dir, smd_gate_scores, run_report)
    _run_generated_figures(repo, results_dir, one_step_scores, run_report)
    _collate_paper_outputs(results_dir)
    verification = _verify(results_dir)
    manifest = _write_manifest(results_dir)
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "commands": run_report,
        "manifest_count": len(manifest),
        "paper_verification": verification,
        "strict": bool(args.strict),
    }
    (results_dir / "capsule_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    missing = [item for item in expected_outputs() if not (results_dir / item).exists()]
    if missing:
        message = "Missing expected capsule outputs: " + ", ".join(missing)
        if args.strict:
            raise SystemExit(message)
        print("WARNING:", message)
    if args.strict and not verification["passed"]:
        raise SystemExit("Paper result verification failed; see capsule_verification.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _prepare_results(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for child in results_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)


def _require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required capsule data directory is missing: {path}")
    return path


def _run(cmd: list[str], cwd: Path, report: list[dict[str, object]], name: str) -> None:
    started = time.perf_counter()
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)
    report.append({"name": name, "elapsed_seconds": round(time.perf_counter() - started, 3), "command": cmd})


def _run_negative_pilots(repo: Path, results_dir: Path, smd_gate_scores: Path, report: list[dict[str, object]]) -> None:
    commands = [
        (
            "rrp_detector",
            [
                sys.executable,
                "-m",
                "hbpc.rrp_detector_experiments",
                "--scores-root",
                str(smd_gate_scores),
                "--output-dir",
                str(results_dir / "results-rrp-detector-pilot"),
                "--seeds",
                *SEEDS,
            ],
        ),
        (
            "mars",
            [
                sys.executable,
                "-m",
                "hbpc.mars_experiments",
                "--scores-root",
                str(smd_gate_scores),
                "--output-dir",
                str(results_dir / "results-mars-smd-pilot"),
                "--seeds",
                *SEEDS,
            ],
        ),
        (
            "ceres",
            [
                sys.executable,
                "-m",
                "hbpc.ceres_experiments",
                "--scores-root",
                str(smd_gate_scores),
                "--output-dir",
                str(results_dir / "results-ceres-smd-pilot"),
                "--seeds",
                *SEEDS,
            ],
        ),
        (
            "scar",
            [
                sys.executable,
                "-m",
                "hbpc.scar_experiments",
                "--scores-root",
                str(smd_gate_scores),
                "--output-dir",
                str(results_dir / "results-scar-smd-pilot"),
                "--seeds",
                *SEEDS,
            ],
        ),
    ]
    for name, cmd in commands:
        _run(cmd, repo, report, name)


def _run_generated_figures(repo: Path, results_dir: Path, one_step_scores: Path, report: list[dict[str, object]]) -> None:
    generated = results_dir / "generated-paper-figures"
    _run(
        [
            sys.executable,
            "scripts/generate_mechanism_figures.py",
            "--output-dir",
            str(generated),
        ],
        repo,
        report,
        "mechanism_figures",
    )
    _run(
        [
            sys.executable,
            "scripts/generate_case_comparison.py",
            "--scores-path",
            str(one_step_scores / "SMD" / "one_step" / "0" / "scores.npz"),
            "--output-dir",
            str(generated),
            "--k",
            "3",
            "--top-n",
            "300",
        ],
        repo,
        report,
        "case_comparison",
    )


def _collate_paper_outputs(results_dir: Path) -> None:
    figures = results_dir / "figures"
    tables = results_dir / "tables"
    metrics = results_dir / "metrics"
    _copy(results_dir / "generated-paper-figures" / "mechanism_relaxation_toy.png", figures / "mechanism_relaxation_toy.png")
    _copy(results_dir / "generated-paper-figures" / "mechanism_forward_average.png", figures / "mechanism_forward_average.png")
    _copy(results_dir / "generated-paper-figures" / "case_comparison_smd_one_step.png", figures / "case_comparison_smd_one_step.png")
    _copy(results_dir / "results-strr-five-dataset" / "figures" / "cross_dataset_rank_biserial.png", figures / "cross_dataset_rank_biserial.png")
    _copy(results_dir / "results-adaptation-causal-score-artifact" / "figures" / "adaptation_tau_vs_r.png", figures / "tau_norm_vs_r.png")
    _copy(results_dir / "results-strr-five-dataset" / "figures" / "budget_curve_smd_one_step.png", figures / "budget_curve_smd_one_step.png")
    _copy(results_dir / "results-strr-alignment" / "figures" / "window_sensitivity_smd.png", figures / "window_sensitivity_smd.png")
    for k in (3, 5, 10, 20):
        _copy(results_dir / "results-rrp-smd-phenomenon" / "figures" / f"relax_ecdf_K{k}.png", figures / f"relax_ecdf_K{k}.png")
        _copy(
            results_dir / "results-rrp-smd-phenomenon" / "figures" / f"relaxation_curve_K{k}.png",
            figures / f"relaxation_curve_K{k}.png",
        )
    for method in PUBLIC_DEEP_METHODS:
        _copy(
            results_dir / "results-public-deep-smd" / "strr" / "figures" / f"budget_curve_smd_{method}.png",
            figures / f"budget_curve_smd_{method}.png",
        )
    _copy(results_dir / "results-synthetic-regime" / "figures" / "synthetic_delta_gain_heatmap.png", figures / "synthetic_delta_gain_heatmap.png")
    _copy(results_dir / "results-uncertainty" / "figures" / "rank_biserial_ci_forest.png", figures / "rank_biserial_ci_forest.png")
    _copy(results_dir / "results-sensitivity" / "figures" / "sensitivity_k_topn_curves.png", figures / "sensitivity_k_topn_curves.png")
    _copy(results_dir / "results-swat-filter" / "figures" / "swat_highpass_regime_movement.png", figures / "swat_highpass_regime_movement.png")

    _copy(results_dir / "results-strr-five-dataset" / "tables" / "cross_dataset_phenomenon.csv", tables / "cross_dataset_phenomenon.csv")
    _copy(results_dir / "results-strr-five-dataset" / "tables" / "budget_curve_summary.csv", tables / "budget_curve_summary.csv")
    _copy(results_dir / "results-strr-five-dataset" / "tables" / "delay_fairness.csv", tables / "delay_fairness.csv")
    _copy(results_dir / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_summary.csv", tables / "adaptation_dataset_summary.csv")
    _copy(
        results_dir / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_correlation_summary.csv",
        tables / "adaptation_dataset_correlation_summary.csv",
    )
    _copy(results_dir / "results-strr-alignment" / "tables" / "postprocess_best_by_predictor.csv", tables / "smd_score_sources.csv")
    _copy(
        results_dir / "results-public-deep-smd" / "strr" / "tables" / "budget_curve_summary.csv",
        tables / "public_deep_budget_curve_summary.csv",
    )
    _write_public_deep_best_mean_std(
        results_dir / "results-public-deep-smd" / "strr" / "metrics" / "supplement_all_rows.csv",
        tables / "public_deep_best_mean_std.csv",
    )
    _copy(results_dir / "results-synthetic-regime" / "tables" / "synthetic_regime_summary.csv", tables / "synthetic_regime_summary.csv")
    _copy(results_dir / "results-synthetic-regime" / "tables" / "synthetic_delta_gain_heatmap.csv", tables / "synthetic_delta_gain_heatmap.csv")
    _write_synthetic_delta_summary(
        results_dir / "results-synthetic-regime" / "tables" / "synthetic_regime_summary.csv",
        results_dir / "results-synthetic-regime" / "tables" / "synthetic_delta_summary.csv",
    )
    _copy(results_dir / "results-synthetic-regime" / "tables" / "synthetic_delta_summary.csv", tables / "synthetic_delta_summary.csv")
    _copy(results_dir / "results-uncertainty" / "tables" / "rank_biserial_uncertainty.csv", tables / "rank_biserial_uncertainty.csv")
    _copy(results_dir / "results-uncertainty" / "tables" / "tau_uncertainty.csv", tables / "tau_uncertainty.csv")
    _copy(results_dir / "results-uncertainty" / "tables" / "correlation_leave_one_out.csv", tables / "correlation_leave_one_out.csv")
    _copy(results_dir / "results-sensitivity" / "tables" / "sensitivity_summary.csv", tables / "sensitivity_summary.csv")
    _copy(results_dir / "results-swat-filter" / "tables" / "swat_highpass_summary.csv", tables / "swat_highpass_summary.csv")
    _write_notation_table(tables / "notation.csv")
    _write_scope_table(tables / "scope.csv")
    _write_regime_summary(results_dir, tables / "regime_summary.csv")
    _write_negative_table(results_dir, tables / "negative_variants.csv")
    _write_numeric_report(results_dir, metrics / "capsule_numeric_report.json")


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_synthetic_delta_summary(src: Path, dst: Path) -> None:
    frame = pd.read_csv(src)
    groups = [
        ("positive", frame[frame["delta_mu"] > 0.1]),
        ("near_equal", frame[frame["delta_mu"].abs() <= 0.1]),
        ("negative", frame[frame["delta_mu"] < -0.1]),
    ]
    rows = []
    for regime, subset in groups:
        rows.append(
            {
                "regime": regime,
                "mean_forward_gain": float(subset["forward_gain"].mean()),
                "mean_raw_f1": float(subset["raw_f1_raw"].mean()),
                "mean_rank_biserial": float(subset["rank_biserial"].mean()),
                "n": int(len(subset)),
            }
        )
    pd.DataFrame(rows).to_csv(dst, index=False)


def _write_public_deep_best_mean_std(src: Path, dst: Path) -> None:
    frame = pd.read_csv(src)
    metric_cols = ["raw_f1", "pa_f1", "event_recall", "mttd"]
    rows: list[dict[str, object]] = []
    for predictor in sorted(frame["predictor"].unique()):
        subset = frame[(frame["dataset"] == "SMD") & (frame["predictor"] == predictor)]
        for postprocess in ("raw", "ewma", "forward_avg"):
            post = subset[subset["postprocess"] == postprocess].copy()
            if post.empty:
                continue
            grouped = (
                post.groupby(["top_n", "k"], as_index=False)[metric_cols]
                .mean()
                .sort_values("raw_f1", ascending=False)
            )
            best = grouped.iloc[0]
            same_setting = post[(post["top_n"] == best["top_n"]) & (post["k"] == best["k"])]
            row = {
                "predictor": predictor,
                "postprocess": postprocess,
                "top_n": int(best["top_n"]),
                "k": int(best["k"]),
            }
            for col in metric_cols:
                row[f"{col}_mean"] = float(same_setting[col].mean())
                row[f"{col}_std"] = float(same_setting[col].std(ddof=1))
            rows.append(row)
    pd.DataFrame(rows).to_csv(dst, index=False)


def _write_notation_table(path: Path) -> None:
    pd.DataFrame(
        [
            {"symbol": "t", "meaning": "time index"},
            {"symbol": "s_t", "meaning": "scalar anomaly score at time t"},
            {"symbol": "y_t", "meaning": "binary anomaly label"},
            {"symbol": "K", "meaning": "short forward horizon"},
            {"symbol": "N", "meaning": "fixed alarm budget in top-N evaluation"},
            {"symbol": "bar_s_t(K)", "meaning": "forward tail mean, K^{-1} sum_{k=1}^{K} s_{t+k}"},
            {"symbol": "rho_t(K)", "meaning": "relaxation ratio, bar_s_t(K)/(s_t + epsilon)"},
            {"symbol": "A", "meaning": "normal-high group: y_t=0 and s_t in the high-score tail"},
            {"symbol": "B", "meaning": "anomaly-high group: y_t=1 and s_t in the high-score tail"},
            {"symbol": "mu_A, mu_B", "meaning": "short-tail locations of groups A and B"},
            {"symbol": "mu_a", "meaning": "tail-location contrast, proportional to mu_B - mu_A"},
            {"symbol": "r", "meaning": "rank-biserial effect size comparing B with A"},
            {"symbol": "tau_N", "meaning": "recovery time constant of normal high-score events"},
            {"symbol": "tau_A", "meaning": "recovery time constant within anomaly segments"},
            {"symbol": "PA-F1", "meaning": "point-adjusted F1"},
            {"symbol": "mTTD", "meaning": "mean time-to-detect, computed from actual alarm times"},
        ]
    ).to_csv(path, index=False)


def _write_scope_table(path: Path) -> None:
    pd.DataFrame(
        [
            {"item": "Input", "scope": "A scalar anomaly score sequence s_t; no raw multivariate state is used."},
            {"item": "Tail window", "scope": "A short post-peak tail s_{t+1:t+K} with bounded confirmation delay."},
            {"item": "Conditioning", "scope": "Timestamps are conditioned on being high-score events before tails are compared."},
            {"item": "Approximation", "scope": "The analytic result uses a location-shift family for the short tail; Gaussian iid noise is the minimum tractable case."},
            {"item": "Ignored structure", "scope": "Seasonality, control inputs, multiscale duration, nonlinear shape, and domain state variables are outside the scalar score-only model."},
            {"item": "Conclusion", "scope": "Sufficiency applies to the short-tail location signal and does not rule out methods that use other information sources."},
        ]
    ).to_csv(path, index=False)


def _write_regime_summary(results_dir: Path, path: Path) -> None:
    phenomenon = pd.read_csv(results_dir / "results-strr-five-dataset" / "tables" / "cross_dataset_phenomenon.csv")
    budget = pd.read_csv(results_dir / "results-strr-five-dataset" / "tables" / "budget_curve_summary.csv")
    adaptation = pd.read_csv(results_dir / "results-adaptation-causal-score-artifact" / "tables" / "adaptation_dataset_summary.csv")
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        r_row = phenomenon[(phenomenon["dataset"] == dataset) & (phenomenon["seed"] == -1) & (phenomenon["k"] == 3)].iloc[0]
        raw = budget[(budget["dataset"] == dataset) & (budget["top_n"] == 300) & (budget["postprocess"] == "raw")].iloc[0]
        fwd = budget[(budget["dataset"] == dataset) & (budget["top_n"] == 300) & (budget["postprocess"] == "forward_avg") & (budget["k"] == 3)].iloc[0]
        tau = adaptation[adaptation["dataset"] == dataset].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "r_K3": r_row["rank_biserial"],
                "tau_anomaly_median": tau["tau_anomaly_median"],
                "tau_normal_median": tau["tau_normal_median"],
                "tau_ratio": tau["tau_ratio"],
                "raw_f1_raw": raw["raw_f1"],
                "raw_f1_forward_K3": fwd["raw_f1"],
                "pa_f1_raw": raw["pa_f1"],
                "pa_f1_forward_K3": fwd["pa_f1"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_negative_table(results_dir: Path, path: Path) -> None:
    frames: list[pd.DataFrame] = []
    sources = {
        "rrp": results_dir / "results-rrp-detector-pilot" / "tables" / "rrp_detector_best.csv",
        "mars": results_dir / "results-mars-smd-pilot" / "tables" / "mars_best.csv",
        "ceres": results_dir / "results-ceres-smd-pilot" / "tables" / "ceres_best_by_family.csv",
        "scar": results_dir / "results-scar-smd-pilot" / "tables" / "scar_best_by_family.csv",
    }
    for source, csv_path in sources.items():
        frame = pd.read_csv(csv_path)
        frame.insert(0, "source", source)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _write_numeric_report(results_dir: Path, path: Path) -> None:
    report = {
        "phenomenon_rows": len(pd.read_csv(results_dir / "tables" / "cross_dataset_phenomenon.csv")),
        "fixed_budget_rows": len(pd.read_csv(results_dir / "tables" / "budget_curve_summary.csv")),
        "public_deep_rows": len(pd.read_csv(results_dir / "tables" / "public_deep_budget_curve_summary.csv")),
        "negative_rows": len(pd.read_csv(results_dir / "tables" / "negative_variants.csv")),
        "synthetic_rows": len(pd.read_csv(results_dir / "tables" / "synthetic_regime_summary.csv")),
        "uncertainty_rows": len(pd.read_csv(results_dir / "tables" / "rank_biserial_uncertainty.csv")),
        "sensitivity_rows": len(pd.read_csv(results_dir / "tables" / "sensitivity_summary.csv")),
        "swat_highpass_rows": len(pd.read_csv(results_dir / "tables" / "swat_highpass_summary.csv")),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _verify(results_dir: Path) -> dict[str, object]:
    report = verify_paper_results(
        results_dir,
        required_groups=(
            "phenomenon",
            "fixed_budget",
            "delay_fair",
            "adaptation",
            "correlation",
            "public_deep",
            "negative",
            "synthetic",
            "uncertainty",
            "sensitivity",
            "swat_highpass",
        ),
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    (results_dir / "metrics" / "paper_result_verification.json").write_text(payload + "\n", encoding="utf-8")
    (results_dir / "paper_result_verification.json").write_text(payload + "\n", encoding="utf-8")
    return report


def _write_manifest(results_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(p for p in results_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(results_dir).as_posix()
        rows.append({"path": rel, "bytes": path.stat().st_size})
    pd.DataFrame(rows).to_csv(results_dir / "capsule_manifest.csv", index=False)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
