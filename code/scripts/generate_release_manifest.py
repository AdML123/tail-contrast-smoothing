from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


BINARY_SUFFIXES = {".pdf", ".png", ".tif", ".tiff"}


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in BINARY_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def public_paths(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    relative_paths = [
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw and raw.decode("utf-8") != "MANIFEST.sha256"
    ]
    return sorted(relative_paths, key=lambda path: path.as_posix())


def generate_manifest(root: Path) -> str:
    rows = []
    for relative in public_paths(root):
        digest = hashlib.sha256(canonical_bytes(root / relative)).hexdigest()
        rows.append(f"{digest}  {relative.as_posix()}")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a cross-platform SHA-256 release manifest."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    (root / "MANIFEST.sha256").write_text(
        generate_manifest(root),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
