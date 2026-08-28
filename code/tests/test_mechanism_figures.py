from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_mechanism_script():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_mechanism_figures.py"
    spec = importlib.util.spec_from_file_location("generate_mechanism_figures", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relaxation_figure_legend_labels_are_compact() -> None:
    module = _load_mechanism_script()

    labels = module.RELAXATION_LEGEND_LABELS

    assert labels == ("normal transient", "persistent anomaly", "short tail K=3")
    assert all(len(label) <= 20 for label in labels)


def test_generate_mechanism_figures_cli_respects_output_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "generate_mechanism_figures.py"

    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "mechanism_relaxation_toy.png").is_file()
    assert (tmp_path / "mechanism_forward_average.png").is_file()
