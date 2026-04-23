from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@dataclass(frozen=True)
class ImageVariant:
    name: str
    path: Path


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for image preprocessing. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return cv2


def read_image(path: Path) -> np.ndarray:
    """Read image using a Windows-friendly unicode path flow."""
    cv2 = _import_cv2()
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    cv2 = _import_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def build_variants(
    image_path: Path,
    output_dir: Path | None = None,
    min_width: int = 1800,
    max_width: int = 2200,
) -> list[ImageVariant]:
    """Create OCR-oriented variants and return their file paths.

    EasyOCR often benefits from both a contrast-enhanced grayscale image and a
    thresholded image. The OCR step later chooses the variant with the stronger
    aggregate confidence.
    """
    cv2 = _import_cv2()
    image = read_image(image_path)

    height, width = image.shape[:2]
    if width < min_width:
        scale = min_width / max(width, 1)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    elif width > max_width:
        scale = max_width / width
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(contrast, 1.6, blurred, -0.6, 0)

    threshold = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    if output_dir is None:
        temp = TemporaryDirectory(prefix="sts_fio_")
        output_dir = Path(temp.name)
        # Keep temp alive by attaching it to the function object for the process
        # lifetime. This is intentionally small and avoids leaking debug files.
        holders = getattr(build_variants, "_temp_holders", [])
        holders.append(temp)
        setattr(build_variants, "_temp_holders", holders)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    variants = [
        ImageVariant("original", image_path),
        ImageVariant("contrast", output_dir / f"{stem}.contrast.png"),
        ImageVariant("threshold", output_dir / f"{stem}.threshold.png"),
    ]
    write_image(variants[1].path, sharpened)
    write_image(variants[2].path, threshold)
    return variants


def build_owner_crop_variant(
    image_path: Path,
    output_dir: Path | None = None,
    min_width: int = 1200,
) -> ImageVariant:
    """Create a focused OCR crop around the STS owner name block."""
    return _build_owner_crop_variant(
        image_path=image_path,
        output_dir=output_dir,
        name="owner_crop",
        crop_box=(0.04, 0.20, 0.62, 0.47),
        min_width=min_width,
    )


def build_owner_crop_variants(
    image_path: Path,
    output_dir: Path | None = None,
) -> list[ImageVariant]:
    """Create broad and narrow OCR crops for the owner block."""
    return [
        build_owner_crop_variant(image_path=image_path, output_dir=output_dir),
        _build_owner_crop_variant(
            image_path=image_path,
            output_dir=output_dir,
            name="owner_name_crop",
            crop_box=(0.10, 0.27, 0.54, 0.44),
            min_width=1200,
        ),
    ]


def _build_owner_crop_variant(
    image_path: Path,
    output_dir: Path | None,
    name: str,
    crop_box: tuple[float, float, float, float],
    min_width: int,
) -> ImageVariant:
    cv2 = _import_cv2()
    image = read_image(image_path)
    height, width = image.shape[:2]

    left, top, right, bottom = crop_box
    x1 = int(width * left)
    x2 = int(width * right)
    y1 = int(height * top)
    y2 = int(height * bottom)
    crop = image[y1:y2, x1:x2]

    crop_height, crop_width = crop.shape[:2]
    if crop_width < min_width:
        scale = min_width / max(crop_width, 1)
        crop = cv2.resize(
            crop,
            (int(crop_width * scale), int(crop_height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=6)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(contrast, 1.8, blurred, -0.8, 0)

    if output_dir is None:
        temp = TemporaryDirectory(prefix="sts_fio_owner_")
        output_dir = Path(temp.name)
        holders = getattr(_build_owner_crop_variant, "_temp_holders", [])
        holders.append(temp)
        setattr(_build_owner_crop_variant, "_temp_holders", holders)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    variant = ImageVariant(name, output_dir / f"{image_path.stem}.{name}.png")
    write_image(variant.path, sharpened)
    return variant
