from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ["SMD", "PSM", "HAI", "MSL", "SMAP", "SWaT"]


def _number(value: object) -> str:
    return "--" if pd.isna(value) else f"{float(value):.3f}"


def _count(value: object) -> str:
    return "--" if pd.isna(value) else str(int(value))


def _text(value: object) -> str:
    return str(value).replace("_", r"\_")


def _predicted_regime(row: pd.Series) -> str:
    if str(row["evidence_status"]) != "primary":
        return "unresolved"
    return {
        "positive": "positive",
        "null_or_uncertain": "null-compatible",
        "reversal": "reversal",
    }.get(str(row["regime"]), "unresolved")


def _evidence_label(value: object) -> str:
    return "P" if str(value) == "primary" else "E"


def generate_spl_tables(input_root: Path | str, output_dir: Path | str) -> list[Path]:
    input_root, output_dir = Path(input_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrast = pd.read_csv(input_root / "tables" / "dataset_tail_contrast.csv")
    performance = pd.read_csv(input_root / "tables" / "delay_aware_performance.csv")

    missing = set(DATASETS) - set(contrast["dataset"].astype(str))
    if missing:
        raise ValueError(f"Table I is missing datasets: {sorted(missing)}")
    required_columns = {
        "matched_pairs",
        "tail_contrast",
        "contrast_ci_low",
        "contrast_ci_high",
        "regime",
        "evidence_status",
    }
    if not required_columns <= set(contrast.columns):
        raise ValueError(
            f"Table I is missing columns: {sorted(required_columns - set(contrast.columns))}"
        )

    gain = performance.pivot_table(
        index="dataset", columns="method", values="raw_f1", aggfunc="mean"
    )
    required_methods = {"raw_realtime", "raw_delayed", "confirmation_mean", "ewma"}
    if not required_methods <= set(gain.columns):
        raise ValueError(f"performance table is missing methods: {sorted(required_methods - set(gain.columns))}")

    rows1: list[str] = []
    for dataset in DATASETS:
        row = contrast[contrast["dataset"] == dataset].iloc[0]
        if dataset not in gain.index:
            raise ValueError(f"Table I is missing performance for {dataset}")
        delta_f1 = gain.loc[dataset, "confirmation_mean"] - gain.loc[dataset, "raw_delayed"]
        rows1.append(
            f"{_text(dataset)} & {_count(row['matched_pairs'])} & {_number(row['tail_contrast'])} & "
            f"[{_number(row['contrast_ci_low'])}, {_number(row['contrast_ci_high'])}] & "
            f"{_text(_predicted_regime(row))} & {_number(delta_f1)} & "
            f"{_evidence_label(row['evidence_status'])} \\\\"
        )
    table1 = (
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{1.0pt}\n"
        "\\begin{tabular}{lrrllrl}\n"
        "\\toprule\n"
        "Dataset & $m$ & $\\delta_\\mu$ & 95\\% CI & Pred. & $\\Delta F1$ & Tier \\\\\n"
        "\\midrule\n"
        + "\n".join(rows1)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )

    confirmation = performance[performance["method"] == "confirmation_mean"].set_index("dataset")
    rows2: list[str] = []
    for dataset in DATASETS:
        if dataset not in gain.index or dataset not in confirmation.index:
            raise ValueError(f"Table II is missing dataset: {dataset}")
        row = confirmation.loc[dataset]
        rows2.append(
            f"{_text(dataset)} & {_number(gain.loc[dataset, 'raw_realtime'])} & "
            f"{_number(gain.loc[dataset, 'raw_delayed'])} & "
            f"{_number(gain.loc[dataset, 'confirmation_mean'])} & "
            f"{_number(gain.loc[dataset, 'ewma'])} & {_number(row['event_recall'])} & "
            f"{_number(row['mttd'])} \\\\"
        )
    table2 = (
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{1.8pt}\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Dataset & \\shortstack{$F1$\\\\raw} & \\shortstack{$F1$\\\\del.} & "
        "\\shortstack{$F1$\\\\conf.} & \\shortstack{$F1$\\\\EWMA} & "
        "\\shortstack{event\\\\recall} & MTTD \\\\\n"
        "\\midrule\n"
        + "\n".join(rows2)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )

    out1 = output_dir / "table1_regimes.tex"
    out2 = output_dir / "table2_performance.tex"
    out1.write_text(table1, encoding="utf-8")
    out2.write_text(table2, encoding="utf-8")
    return [out1, out2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    generate_spl_tables(args.input_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
