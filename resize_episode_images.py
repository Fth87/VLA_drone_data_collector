"""Resize all episode images to 224x224.

Default behavior:
- reads images from dataset/videos
- writes resized images to dataset/videos_224
- preserves the episode folder structure
- center-crops to square before resizing to avoid distortion

Usage:
    uv run python resize_episode_images.py
    uv run python resize_episode_images.py --input-dir dataset/videos --output-dir dataset/videos_224 --size 224
    uv run python resize_episode_images.py --in-place
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image

    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def resize_image(source_path: Path, target_path: Path, size: int) -> None:
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB") if image.mode not in ("RGB", "L") else image
        image = center_crop_square(image)
        image = image.resize((size, size), Image.Resampling.LANCZOS)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = target_path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            image.save(target_path, quality=85, optimize=True)
        elif suffix == ".webp":
            image.save(target_path, quality=85, method=6)
        else:
            image.save(target_path, optimize=True, compress_level=9)


def iter_image_files(root_dir: Path):
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_target_path(source_path: Path, input_dir: Path, output_dir: Path, in_place: bool) -> Path:
    if in_place:
        return source_path

    relative_path = source_path.relative_to(input_dir)
    return output_dir / relative_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resize episode images to 224x224.")
    parser.add_argument("--input-dir", type=Path, default=Path("dataset/videos"), help="Source image root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/videos_224"), help="Output root directory")
    parser.add_argument("--size", type=int, default=224, help="Target square size")
    parser.add_argument("--in-place", action="store_true", help="Overwrite images in the input directory")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    processed = 0
    for source_path in iter_image_files(input_dir):
        target_path = build_target_path(source_path, input_dir, output_dir, args.in_place)
        resize_image(source_path, target_path, args.size)
        processed += 1

    print(f"Processed {processed} image(s) to {args.size}x{args.size}.")
    if not args.in_place:
        print(f"Output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
