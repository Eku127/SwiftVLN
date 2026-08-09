"""Shared image helpers for SatDronePair generation and post-processing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image as an independent RGB image and close the source file."""
    with Image.open(path) as image:
        return image.convert("RGB")


def center_square_recrop(
    image: Image.Image,
    crop_size: int,
    output_size: int,
) -> Image.Image:
    """Center-crop a square region and resize it to ``output_size``."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    if crop_size > image.width or crop_size > image.height:
        raise ValueError(
            f"crop_size={crop_size} exceeds image size {image.size}"
        )

    left = (image.width - crop_size) // 2
    top = (image.height - crop_size) // 2
    return image.crop(
        (left, top, left + crop_size, top + crop_size)
    ).resize((output_size, output_size), Image.Resampling.LANCZOS)
