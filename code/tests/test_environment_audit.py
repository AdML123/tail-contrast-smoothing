from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_environment import build_audit_manifest


def test_audit_manifest_records_provenance_and_hashes(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    result_root.mkdir()
    artifact = result_root / "metrics.json"
    artifact.write_text(json.dumps({"score": 1.0}), encoding="utf-8")

    manifest = build_audit_manifest(
        result_root=result_root,
        command=["conda", "run", "-n", "paper47-spl", "python", "script.py", "--strict"],
        start_time="2026-08-28T00:00:00+08:00",
        end_time="2026-08-28T00:01:00+08:00",
        exit_status=0,
        git_commit="abc1234",
        conda_version="conda 25.7.0",
    )

    required = {
        "git_commit",
        "platform",
        "conda_version",
        "python_version",
        "resolved_packages",
        "command",
        "start_time",
        "end_time",
        "duration_seconds",
        "exit_status",
        "cuda_available",
        "result_root",
        "output_hashes",
    }
    assert required <= manifest.keys()
    assert manifest["command"] == ["conda", "run", "-n", "paper47-spl", "python", "script.py", "--strict"]
    assert manifest["exit_status"] == 0
    assert manifest["duration_seconds"] == 60.0
    assert manifest["output_hashes"]["metrics.json"]
    assert manifest["result_root"] == str(result_root)
