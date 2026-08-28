"""Record reproducibility metadata for a capsule run without changing its outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


def _duration_seconds(start_time: str, end_time: str) -> float:
    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)
    return round((end - start).total_seconds(), 6)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_hashes(result_root: Path) -> dict[str, str]:
    if not result_root.exists():
        return {}
    return {
        str(path.relative_to(result_root)).replace("\\", "/"): _sha256(path)
        for path in sorted(result_root.rglob("*"))
        if path.is_file()
    }


def build_audit_manifest(
    *,
    result_root: Path,
    command: Sequence[str],
    start_time: str,
    end_time: str,
    exit_status: int,
    git_commit: str,
    conda_version: str,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> dict[str, object]:
    """Build a JSON-serializable run manifest from read-only observations."""
    manifest: dict[str, object] = {
        "git_commit": git_commit,
        "platform": platform.platform(),
        "conda_version": conda_version,
        "python_version": sys.version,
        "resolved_packages": _package_versions(),
        "command": list(command),
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": _duration_seconds(start_time, end_time),
        "exit_status": int(exit_status),
        "cuda_available": _cuda_available(),
        "result_root": str(result_root),
        "output_hashes": _output_hashes(result_root),
    }
    if stdout_path is not None:
        manifest["stdout_path"] = str(stdout_path)
    if stderr_path is not None:
        manifest["stderr_path"] = str(stderr_path)
    return manifest


def _command_output(command: Iterable[str]) -> str:
    try:
        return subprocess.run(list(command), check=False, capture_output=True, text=True).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--conda-version", default="unknown")
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--stdout-path", type=Path)
    parser.add_argument("--stderr-path", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    manifest = build_audit_manifest(
        result_root=args.result_root.resolve(),
        command=command,
        start_time=args.start_time,
        end_time=args.end_time,
        exit_status=args.exit_status,
        git_commit=args.git_commit,
        conda_version=args.conda_version,
        stdout_path=args.stdout_path,
        stderr_path=args.stderr_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
