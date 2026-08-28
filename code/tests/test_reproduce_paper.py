from pathlib import Path
import subprocess
import sys


def test_reproduce_paper_dry_run_can_include_public_deep():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper.py",
            "--skip-download",
            "--dry-run",
            "--include-public-deep",
            "--anomaly-transformer-root",
            "external/Anomaly-Transformer",
            "--tsl-root",
            "external/Time-Series-Library",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "export_anomaly_transformer_scores.py" in result.stdout
    assert "export_tsl_public_scores.py" in result.stdout
    assert "results-public-deep-smd" in result.stdout
    assert "--methods TimesNet Transformer Autoformer AnomalyTransformer" in result.stdout
