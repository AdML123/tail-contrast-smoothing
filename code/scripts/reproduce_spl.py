from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hbpc.spl_result_verifier import verify_spl_results


STAGES = ("spl_experiments", "synthetic_regime_experiments", "swat_filter_experiments", "spl_result_verifier")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the corrected SPL evidence path.")
    parser.add_argument("--score-root", type=Path, default=Path("../data/score-artifacts/results-five-dataset-one-step-full/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("../results/spl-final"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        for stage in STAGES:
            print(stage)
        return 0
    output = args.output_root
    commands = [
        [sys.executable, "-m", "hbpc.spl_experiments", "--score-root", str(args.score_root), "--output-dir", str(output / "corrected")],
        [sys.executable, "-m", "hbpc.synthetic_regime_experiments", "--output-dir", str(output / "synthetic")],
        [sys.executable, "-m", "hbpc.swat_filter_experiments", "--score-root", str(args.score_root), "--output-dir", str(output / "swat"), "--windows", "3", "5", "10", "20"],
    ]
    for command in commands:
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return int(completed.returncode)
    report = verify_spl_results(output / "corrected")
    (output / "verification.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
