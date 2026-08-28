"""Generate the compact case table used by the SPL letter."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ORDER = ["positive", "null-compatible", "reversal", "exploratory"]
DATASETS = ["SMD", "PSM", "HAI", "MSL", "SMAP", "SWaT"]


def _num(value: object) -> str:
    return f"{float(value):.3f}"


def generate(input_root: Path, output: Path) -> Path:
    tables = input_root / "tables"
    contrast = pd.read_csv(tables / "dataset_tail_contrast.csv").set_index("dataset")
    performance = pd.read_csv(tables / "delay_aware_performance.csv")
    pivot = performance.pivot_table(index="dataset", columns="method", values="raw_f1", aggfunc="mean")
    missing = set(DATASETS) - set(contrast.index) - set(pivot.index)
    if missing:
        raise ValueError(f"missing datasets: {sorted(missing)}")
    rows: list[str] = []
    for case in ORDER:
        for dataset in DATASETS:
            row = contrast.loc[dataset]
            status = str(row["evidence_status"])
            regime = str(row["regime"])
            if case == "positive" and regime != "positive":
                continue
            if case == "null-compatible" and regime != "null_or_uncertain":
                continue
            if case == "reversal" and regime != "reversal":
                continue
            if case == "exploratory" and status != "exploratory_underpowered":
                continue
            label = "null-compatible" if case == "null-compatible" else case
            gain = pivot.loc[dataset, "confirmation_mean"] - pivot.loc[dataset, "raw_delayed"]
            rows.append(
                f"{label} & {dataset} & {int(row['matched_pairs'])} & "
                f"{_num(row['tail_contrast'])} & [{_num(row['contrast_ci_low'])}, {_num(row['contrast_ci_high'])}] & "
                f"{_num(gain)} \\\\"
            )
    text = (
        "\\footnotesize\n\\setlength{\\tabcolsep}{1.55pt}\n"
        "\\begin{tabular}{llr l r l}\n\\toprule\n"
        "Case & Dataset & $m$ & $\\delta_\\mu$ & 95\\% CI & $\\Delta F1$ \\\\\n"
        "\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.input_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
