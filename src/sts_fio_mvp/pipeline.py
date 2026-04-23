from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .extractor import ExtractorMode, FioResult, create_extractor
from .ocr import EasyOcrEngine, OcrResult
from .postprocess import refine_fio_from_ocr
from .preprocess import ImageVariant, build_owner_crop_variants, build_variants


@dataclass(frozen=True)
class StsExtractionResult:
    file: str
    ocr: dict
    fio: dict


def process_image(
    image_path: Path,
    ocr_engine: EasyOcrEngine,
    extractor,
    debug_dir: Path | None = None,
    ocr_variant_names: tuple[str, ...] = ("contrast",),
    use_owner_crop: bool = True,
    progress: Callable[[str], None] | None = None,
) -> StsExtractionResult:
    variants_dir = debug_dir / image_path.stem if debug_dir else None
    if progress:
        progress("Preparing OCR image variants")
    variants = build_variants(image_path, output_dir=variants_dir)
    variants = _select_variants(variants, ocr_variant_names)
    ocr_result: OcrResult = ocr_engine.recognize_best(variants, progress=progress)
    if use_owner_crop:
        ocr_result = _with_owner_crop_ocr(
            image_path=image_path,
            ocr_engine=ocr_engine,
            base_result=ocr_result,
            debug_dir=variants_dir,
            progress=progress,
        )
    if progress:
        progress("Extracting FIO with BERT NER")
    raw_fio_result: FioResult = extractor.extract(ocr_result.text)
    fio_result = refine_fio_from_ocr(ocr_result, raw_fio_result)

    return _build_result(image_path=image_path, ocr_result=ocr_result, fio_result=fio_result)


def _build_result(
    image_path: Path,
    ocr_result: OcrResult,
    fio_result: FioResult,
) -> StsExtractionResult:
    return StsExtractionResult(
        file=str(image_path),
        ocr={
            "text": ocr_result.text,
            "variant": ocr_result.variant,
            "mean_confidence": ocr_result.mean_confidence,
            "items": [asdict(item) for item in ocr_result.items],
        },
        fio=fio_result.to_dict(),
    )


def _with_owner_crop_ocr(
    image_path: Path,
    ocr_engine: EasyOcrEngine,
    base_result: OcrResult,
    debug_dir: Path | None,
    progress: Callable[[str], None] | None = None,
) -> OcrResult:
    if progress:
        progress("OCR: reading focused STS owner crop")
    owner_results = [
        ocr_engine.recognize(
            owner_variant,
            decoder="beamsearch",
            allowlist=_owner_crop_allowlist(),
        )
        for owner_variant in build_owner_crop_variants(image_path, output_dir=debug_dir)
    ]
    text = "\n".join(
        part
        for part in (
            *(owner_result.text for owner_result in owner_results),
            base_result.text,
        )
        if part
    )
    item_count = len(base_result.items) + sum(
        len(owner_result.items) for owner_result in owner_results
    )
    confidence = 0.0
    if item_count:
        confidence = (
            base_result.mean_confidence * len(base_result.items)
            + sum(
                owner_result.mean_confidence * len(owner_result.items)
                for owner_result in owner_results
            )
        ) / item_count
    return OcrResult(
        text=text,
        items=[
            *base_result.items,
            *(item for owner_result in owner_results for item in owner_result.items),
        ],
        variant="+".join(
            [base_result.variant, *(owner_result.variant for owner_result in owner_results)]
        ),
        mean_confidence=confidence,
    )


def _owner_crop_allowlist() -> str:
    cyrillic = "".join(chr(code) for code in range(0x0410, 0x0450))
    latin = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return cyrillic + latin + " -()"


def _select_variants(
    variants: list[ImageVariant],
    names: tuple[str, ...],
) -> list[ImageVariant]:
    selected = [variant for variant in variants if variant.name in names]
    if not selected:
        expected = ", ".join(variant.name for variant in variants)
        requested = ", ".join(names)
        raise ValueError(
            f"Unknown OCR variant requested: {requested}. Available variants: {expected}."
        )
    return selected


def extract_fio_from_image(
    image_path: str | Path,
    *,
    extractor_mode: ExtractorMode | str = ExtractorMode.TOKEN_CLASSIFICATION,
    model_name: str | None = None,
    debug_dir: str | Path | None = None,
    use_gpu: bool = False,
    bert_device: int = -1,
    ocr_variant_names: tuple[str, ...] = ("contrast",),
    use_owner_crop: bool = True,
    progress: Callable[[str], None] | None = None,
) -> StsExtractionResult:
    """Run the full OCR + BERT extraction pipeline for one image.

    This is the main API for manual use from Python code:

    result = extract_fio_from_image("data/test3.png")
    print(result.fio)
    """
    image_path = Path(image_path)
    debug_path = Path(debug_dir) if debug_dir is not None else None
    extractor_mode = ExtractorMode(extractor_mode)

    if progress:
        progress("Initializing EasyOCR")
    ocr_engine = EasyOcrEngine(gpu=use_gpu)
    if progress:
        progress("Preparing OCR image variants")
    variants_dir = debug_path / image_path.stem if debug_path else None
    variants = build_variants(image_path, output_dir=variants_dir)
    variants = _select_variants(variants, ocr_variant_names)

    ocr_result: OcrResult = ocr_engine.recognize_best(variants, progress=progress)
    if use_owner_crop:
        ocr_result = _with_owner_crop_ocr(
            image_path=image_path,
            ocr_engine=ocr_engine,
            base_result=ocr_result,
            debug_dir=variants_dir,
            progress=progress,
        )

    if progress:
        progress("Loading BERT NER model")
    extractor = create_extractor(
        mode=extractor_mode,
        model_name=model_name,
        device=bert_device,
    )
    if progress:
        progress("Extracting FIO with BERT NER")
    raw_fio_result: FioResult = extractor.extract(ocr_result.text)
    fio_result = refine_fio_from_ocr(ocr_result, raw_fio_result)

    if progress:
        progress("Done")
    return _build_result(image_path=image_path, ocr_result=ocr_result, fio_result=fio_result)
