"""OCR + BERT FIO extraction MVP for Russian STS images."""

from .extractor import DEFAULT_NER_MODEL, DEFAULT_QA_MODEL, ExtractorMode, FioResult
from .pipeline import StsExtractionResult, extract_fio_from_image

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_NER_MODEL",
    "DEFAULT_QA_MODEL",
    "ExtractorMode",
    "FioResult",
    "StsExtractionResult",
    "extract_fio_from_image",
]
