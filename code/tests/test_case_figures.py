from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def test_generate_case_comparison_cli_writes_figure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_case_comparison.py"
    scores_path = tmp_path / "scores.npz"
    scores = np.array(
        [1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 7, 6, 6, 1, 1],
        dtype=float,
    )
    labels = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        dtype=int,
    )
    np.savez(scores_path, scores=scores, labels=labels)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--scores-path",
            str(scores_path),
            "--output-dir",
            str(tmp_path),
            "--k",
            "3",
            "--top-n",
            "3",
            "--radius-before",
            "2",
            "--radius-after",
            "5",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "case_comparison_smd_one_step.png").is_file()
