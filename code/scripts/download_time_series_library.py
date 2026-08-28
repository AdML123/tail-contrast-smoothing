"""Download THUML Time-Series-Library from Hugging Face.

Usage:
    python scripts/download_time_series_library.py --local-dir datasets/Time-Series-Library
"""

from __future__ import annotations

import argparse
from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="datasets/Time-Series-Library")
    args = parser.parse_args()
    snapshot_download(
        repo_id="thuml/Time-Series-Library",
        repo_type="dataset",
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded thuml/Time-Series-Library to {args.local_dir}")


if __name__ == "__main__":
    main()
