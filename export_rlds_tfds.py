from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from rlds_tfds_builder import MAX_EPISODES_ENV, SOURCE_DIR_ENV, build_builder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Roblox VLA dataset to TFDS TFRecord shards.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Source dataset root containing data/, meta/, and videos/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_tfds",
        help="TFDS output directory.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional smoke-test limit. Use 0 to export all episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before exporting.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    os.environ[SOURCE_DIR_ENV] = str(source_dir)
    if args.max_episodes > 0:
        os.environ[MAX_EPISODES_ENV] = str(args.max_episodes)
    else:
        os.environ.pop(MAX_EPISODES_ENV, None)

    builder = build_builder(output_dir)
    builder.download_and_prepare(file_format="tfrecord")

    print(f"Exported TFDS dataset to: {builder.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())