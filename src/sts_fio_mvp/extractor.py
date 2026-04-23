from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


DEFAULT_QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"
DEFAULT_NER_MODEL = "Gherman/bert-base-NER-Russian"


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


class ExtractorMode(str, Enum):
    QA = "qa"
    TOKEN_CLASSIFICATION = "token-classification"


def _import_pipeline():
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for BERT extraction. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return pipeline


def _import_qa_dependencies():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Torch and transformers are required for BERT QA extraction. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return torch, AutoModelForQuestionAnswering, AutoTokenizer


def _normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\n", " ")
    text = " ".join(text.split()).strip(" .,:;|-")
    return text or None


class BertQaFioExtractor:
    """Extract FIO fields with a BERT extractive QA model.

    The implementation calls AutoModelForQuestionAnswering directly instead of
    transformers.pipeline("question-answering"). Some transformers builds do not
    register that pipeline task, while the underlying BERT QA model still works.
    """

    QUESTIONS = {
        "surname": (
            "\u041a\u0430\u043a\u0430\u044f \u0444\u0430\u043c\u0438\u043b\u0438\u044f "
            "\u0441\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u0438\u043a\u0430 "
            "\u0438\u043b\u0438 \u0432\u043b\u0430\u0434\u0435\u043b\u044c\u0446\u0430 "
            "\u0442\u0440\u0430\u043d\u0441\u043f\u043e\u0440\u0442\u043d\u043e\u0433\u043e "
            "\u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430?"
        ),
        "name": (
            "\u041a\u0430\u043a\u043e\u0435 \u0438\u043c\u044f "
            "\u0441\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u0438\u043a\u0430 "
            "\u0438\u043b\u0438 \u0432\u043b\u0430\u0434\u0435\u043b\u044c\u0446\u0430 "
            "\u0442\u0440\u0430\u043d\u0441\u043f\u043e\u0440\u0442\u043d\u043e\u0433\u043e "
            "\u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430?"
        ),
        "patronymic": (
            "\u041a\u0430\u043a\u043e\u0435 \u043e\u0442\u0447\u0435\u0441\u0442\u0432\u043e "
            "\u0441\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u0438\u043a\u0430 "
            "\u0438\u043b\u0438 \u0432\u043b\u0430\u0434\u0435\u043b\u044c\u0446\u0430 "
            "\u0442\u0440\u0430\u043d\u0441\u043f\u043e\u0440\u0442\u043d\u043e\u0433\u043e "
            "\u0441\u0440\u0435\u0434\u0441\u0442\u0432\u0430?"
        ),
    }

    def __init__(
        self,
        model_name: str = DEFAULT_QA_MODEL,
        device: int = -1,
        max_answer_len: int = 32,
        max_length: int = 512,
        doc_stride: int = 128,
    ) -> None:
        torch, auto_model, auto_tokenizer = _import_qa_dependencies()
        self._torch = torch
        self._tokenizer = auto_tokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self._tokenizer, "is_fast", False):
            raise RuntimeError(
                "QA extraction requires a fast tokenizer to map BERT tokens back "
                "to OCR text spans."
            )

        self._model = auto_model.from_pretrained(model_name)
        self._device = self._resolve_device(torch, device)
        self._model.to(self._device)
        self._model.eval()

        self._max_answer_len = max_answer_len
        self._max_length = max_length
        self._doc_stride = doc_stride

    @staticmethod
    def _resolve_device(torch_module, device: int):
        if device >= 0 and torch_module.cuda.is_available():
            return torch_module.device(f"cuda:{device}")
        return torch_module.device("cpu")

    def extract(self, text: str) -> FioResult:
        fields: dict[str, ExtractedField] = {}
        for field_name, question in self.QUESTIONS.items():
            answer, score = self._answer(question=question, context=text)
            fields[field_name] = ExtractedField(
                value=answer,
                confidence=score,
                source=ExtractorMode.QA.value,
            )
        return FioResult(
            surname=fields["surname"],
            name=fields["name"],
            patronymic=fields["patronymic"],
        )

    def _answer(self, question: str, context: str) -> tuple[str | None, float]:
        context = context.strip()
        if not context:
            return None, 0.0

        encoded = self._tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self._max_length,
            stride=self._doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)

        model_inputs = {
            name: value.to(self._device)
            for name, value in encoded.items()
            if hasattr(value, "to")
        }

        with self._torch.no_grad():
            outputs = self._model(**model_inputs)

        start_probs = self._torch.softmax(outputs.start_logits, dim=-1)
        end_probs = self._torch.softmax(outputs.end_logits, dim=-1)

        best_text: str | None = None
        best_score = 0.0
        feature_count = int(start_probs.shape[0])

        for feature_index in range(feature_count):
            sequence_ids = encoded.sequence_ids(feature_index)
            context_token_indexes = {
                index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
            }

            start_top = self._top_indexes(start_probs[feature_index])
            end_top = self._top_indexes(end_probs[feature_index])

            for start_index in start_top:
                if start_index not in context_token_indexes:
                    continue
                for end_index in end_top:
                    if end_index not in context_token_indexes:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > self._max_answer_len:
                        continue

                    char_start, _ = offset_mapping[feature_index][start_index].tolist()
                    _, char_end = offset_mapping[feature_index][end_index].tolist()
                    if char_end <= char_start:
                        continue

                    score = (
                        start_probs[feature_index][start_index]
                        * end_probs[feature_index][end_index]
                    ).item()
                    if score <= best_score:
                        continue

                    best_text = context[char_start:char_end]
                    best_score = float(score)

        return _normalize_answer(best_text), best_score

    def _top_indexes(self, probabilities, count: int = 20) -> list[int]:
        count = min(count, int(probabilities.shape[-1]))
        return self._torch.topk(probabilities, k=count).indices.tolist()


class BertTokenClassificationFioExtractor:
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
                    source=ExtractorMode.TOKEN_CLASSIFICATION.value,
                )

        if person_spans and not all(field in chosen for field in ("surname", "name")):
            self._fill_from_person_span(chosen, person_spans)

        empty = ExtractedField(
            value=None,
            confidence=None,
            source=ExtractorMode.TOKEN_CLASSIFICATION.value,
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
                    source=ExtractorMode.TOKEN_CLASSIFICATION.value,
                )


def create_extractor(
    mode: ExtractorMode,
    model_name: str | None = None,
    device: int = -1,
) -> BertQaFioExtractor | BertTokenClassificationFioExtractor:
    if mode == ExtractorMode.QA:
        return BertQaFioExtractor(model_name=model_name or DEFAULT_QA_MODEL, device=device)
    return BertTokenClassificationFioExtractor(
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
