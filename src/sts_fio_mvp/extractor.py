from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_NER_MODEL = "Gherman/bert-base-NER-Russian"
TOKEN_CLASSIFICATION_SOURCE = "token-classification"


@dataclass(frozen=True)
class ExtractedField:
    value: str | None
    confidence: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class FioResult:
    surname: ExtractedField
    name: ExtractedField
    patronymic: ExtractedField

    def to_dict(self) -> dict[str, dict[str, str | float | None]]:
        return {
            "surname": asdict(self.surname),
            "name": asdict(self.name),
            "patronymic": asdict(self.patronymic),
        }


def _import_pipeline():
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for FIO extraction. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return pipeline


def _normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\n", " ")
    text = " ".join(text.split()).strip(" .,:;|-")
    return text or None


class BertFioExtractor:
    """Extract fields with a fine-tuned token-classification BERT model.

    Expected entity labels are SURNAME, NAME and PATRONYMIC. BIO prefixes are
    accepted by the HuggingFace aggregation pipeline and stripped here.
    """

    LABEL_TO_FIELD = {
        "SURNAME": "surname",
        "LASTNAME": "surname",
        "LAST_NAME": "surname",
        "FAMILY": "surname",
        "NAME": "name",
        "FIRSTNAME": "name",
        "FIRST_NAME": "name",
        "PATRONYMIC": "patronymic",
        "MIDDLENAME": "patronymic",
        "MIDDLE_NAME": "patronymic",
    }

    def __init__(self, model_name: str, device: int = -1) -> None:
        pipeline = _import_pipeline()
        self._ner = pipeline(
            "token-classification",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple",
            device=device,
        )

    def extract(self, text: str) -> FioResult:
        entities = self._predict_entities(text)
        chosen: dict[str, ExtractedField] = {}
        person_spans: list[tuple[str, float]] = []

        for entity in entities:
            label = str(entity.get("entity_group") or entity.get("entity") or "")
            label = label.split("-")[-1].upper()
            field = self.LABEL_TO_FIELD.get(label)
            value = _normalize_answer(entity.get("word"))
            confidence = float(entity.get("score", 0.0))

            if label in {"PER", "PERSON"} and value:
                person_spans.append((value, confidence))
                continue

            if not field or not value:
                continue

            current = chosen.get(field)
            if current is None or (current.confidence or 0.0) < confidence:
                chosen[field] = ExtractedField(
                    value=value,
                    confidence=confidence,
                    source=TOKEN_CLASSIFICATION_SOURCE,
                )

        if person_spans and not all(field in chosen for field in ("surname", "name")):
            self._fill_from_person_span(chosen, person_spans)

        empty = ExtractedField(
            value=None,
            confidence=None,
            source=TOKEN_CLASSIFICATION_SOURCE,
        )
        return FioResult(
            surname=chosen.get("surname", empty),
            name=chosen.get("name", empty),
            patronymic=chosen.get("patronymic", empty),
        )

    def _predict_entities(self, text: str) -> list[dict[str, Any]]:
        variants = [text]
        titled = _titlecase_cyrillic_words(text)
        if titled != text:
            variants.append(titled)

        entities: list[dict[str, Any]] = []
        for variant in variants:
            entities.extend(self._ner(variant))
        return entities

    def _fill_from_person_span(
        self,
        chosen: dict[str, ExtractedField],
        person_spans: list[tuple[str, float]],
    ) -> None:
        value, confidence = max(person_spans, key=lambda item: (item[1], len(item[0])))
        parts = _person_name_parts(value)
        for field, part in zip(("surname", "name", "patronymic"), parts):
            if field not in chosen:
                chosen[field] = ExtractedField(
                    value=part,
                    confidence=confidence,
                    source=TOKEN_CLASSIFICATION_SOURCE,
                )


def create_extractor(
    model_name: str | None = None,
    device: int = -1,
) -> BertFioExtractor:
    return BertFioExtractor(
        model_name=model_name or DEFAULT_NER_MODEL,
        device=device,
    )


def _titlecase_cyrillic_words(text: str) -> str:
    chunks: list[str] = []
    current: list[str] = []

    def flush_word() -> None:
        if not current:
            return
        word = "".join(current)
        if _has_cyrillic(word):
            chunks.append(word[:1].upper() + word[1:].lower())
        else:
            chunks.append(word)
        current.clear()

    for char in text:
        if char.isalpha() or char == "-":
            current.append(char)
        else:
            flush_word()
            chunks.append(char)
    flush_word()
    return "".join(chunks)


def _has_cyrillic(value: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in value)


def _person_name_parts(value: str) -> list[str]:
    parts: list[str] = []
    for raw_part in value.replace("\n", " ").split():
        part = raw_part.strip(" .,:;|()[]{}")
        if part and _has_cyrillic(part):
            parts.append(part)
    return parts[:3]
