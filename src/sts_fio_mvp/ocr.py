from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .preprocess import ImageVariant, read_image


@dataclass(frozen=True)
class OcrItem:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class OcrResult:
    text: str
    items: list[OcrItem]
    variant: str
    mean_confidence: float


def _import_easyocr():
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is required for OCR. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return easyocr


class EasyOcrEngine:
    def __init__(
        self,
        languages: tuple[str, ...] = ("ru", "en"),
        gpu: bool = False,
        decoder: str = "greedy",
        batch_size: int = 4,
    ) -> None:
        easyocr = _import_easyocr()
        self._reader = easyocr.Reader(list(languages), gpu=gpu)
        self._decoder = decoder
        self._batch_size = batch_size

    def recognize_best(
        self,
        variants: list[ImageVariant],
        progress: Callable[[str], None] | None = None,
    ) -> OcrResult:
        results = []
        for variant in variants:
            if progress:
                progress(f"OCR: распознаю вариант изображения '{variant.name}'")
            results.append(self.recognize(variant))
        return max(results, key=lambda result: (result.mean_confidence, len(result.text)))

    def recognize(
        self,
        variant: ImageVariant,
        decoder: str | None = None,
        allowlist: str | None = None,
    ) -> OcrResult:
        image = read_image(variant.path)
        readtext_kwargs = {
            "detail": 1,
            "paragraph": False,
            "decoder": decoder or self._decoder,
            "batch_size": self._batch_size,
        }
        if allowlist:
            readtext_kwargs["allowlist"] = allowlist

        raw_items = self._reader.readtext(image, **readtext_kwargs)

        items = [
            OcrItem(
                text=str(text).strip(),
                confidence=float(confidence or 0.0),
                bbox=tuple((float(x), float(y)) for x, y in bbox),
            )
            for bbox, text, confidence in raw_items
            if str(text).strip()
        ]

        text = items_to_text(items)
        confidence = sum(item.confidence for item in items) / len(items) if items else 0.0
        return OcrResult(
            text=text,
            items=items,
            variant=variant.name,
            mean_confidence=confidence,
        )


def items_to_text(items: list[OcrItem]) -> str:
    """Sort OCR fragments into approximate reading order."""
    if not items:
        return ""

    def center_y(item: OcrItem) -> float:
        return sum(point[1] for point in item.bbox) / len(item.bbox)

    def center_x(item: OcrItem) -> float:
        return sum(point[0] for point in item.bbox) / len(item.bbox)

    heights = [
        max(point[1] for point in item.bbox) - min(point[1] for point in item.bbox)
        for item in items
    ]
    row_threshold = max(12.0, (sum(heights) / len(heights)) * 0.65)

    rows: list[list[OcrItem]] = []
    for item in sorted(items, key=lambda part: (center_y(part), center_x(part))):
        y = center_y(item)
        for row in rows:
            row_y = sum(center_y(part) for part in row) / len(row)
            if abs(row_y - y) <= row_threshold:
                row.append(item)
                break
        else:
            rows.append([item])

    lines: list[str] = []
    for row in rows:
        ordered = sorted(row, key=center_x)
        line = " ".join(part.text for part in ordered)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)
