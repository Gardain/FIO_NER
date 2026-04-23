from __future__ import annotations

from dataclasses import replace

from .extractor import ExtractedField, FioResult
from .ocr import OcrResult


FIELD_ORDER = ("surname", "name", "patronymic")


def refine_fio_from_ocr(
    ocr_result: OcrResult,
    fio_result: FioResult,
) -> FioResult:
    """Clean BERT NER output with STS/OCR-specific layout signals."""
    layout_candidates = _owner_name_candidates(ocr_result.text)
    raw_fields = {
        "surname": fio_result.surname,
        "name": fio_result.name,
        "patronymic": fio_result.patronymic,
    }
    cleaned = {
        field: _clean_field(field, extracted)
        for field, extracted in raw_fields.items()
    }

    for index, field in enumerate(FIELD_ORDER):
        if index >= len(layout_candidates):
            continue

        candidate = layout_candidates[index]
        current = cleaned[field]
        cleaned[field] = ExtractedField(
            value=candidate,
            confidence=current.confidence or ocr_result.mean_confidence,
            source=f"{current.source}+ocr-layout" if current.source else "ocr-layout",
        )

    return FioResult(
        surname=cleaned["surname"],
        name=cleaned["name"],
        patronymic=cleaned["patronymic"],
    )


def _clean_field(field: str, extracted: ExtractedField) -> ExtractedField:
    value = extracted.value
    if not value:
        return extracted

    candidate = _name_candidate_from_line(value)
    if candidate:
        if field == "patronymic" and not _looks_like_patronymic(candidate):
            return replace(extracted, value=None)
        return replace(extracted, value=candidate)

    if field == "patronymic" and not _has_cyrillic(value):
        return replace(extracted, value=None)

    if _has_cyrillic(value):
        titled = _titlecase_cyrillic_words(value)
        if field == "patronymic" and not _looks_like_patronymic(titled):
            return replace(extracted, value=None)
        return replace(extracted, value=titled)

    return extracted


def _owner_name_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    windows: list[tuple[int, list[str]]] = [(0, lines[:10])]
    windows.extend((index, lines[index : index + 12]) for index in _owner_label_indexes(lines))

    scored = [
        (score, index, candidates)
        for index, window in windows
        for candidates, score in [_structured_owner_candidates(window)]
        if candidates
    ]
    if not scored:
        return []

    _, _, candidates = max(scored, key=lambda item: (len(item[2]), item[0], -item[1]))
    return candidates


def _structured_owner_candidates(lines: list[str]) -> tuple[list[str], int]:
    candidates: list[str] = []
    score = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if _line_has_owner_label(line):
            score += 5
            continue

        if candidates and _line_is_section_boundary(line):
            break

        latin_duplicate = _latin_duplicate_from_line(line)
        if latin_duplicate and candidates:
            corrected = _correct_from_latin_duplicate(candidates[-1], latin_duplicate)
            if corrected != candidates[-1]:
                candidates[-1] = corrected
                score += 4
            else:
                score += 1
            continue

        candidate = _strict_name_candidate_from_line(line)
        if not candidate:
            if candidates and _looks_like_noise_after_names(line):
                break
            continue

        if candidate not in candidates:
            if len(candidates) == 2 and not _looks_like_patronymic(candidate):
                break
            candidates.append(candidate)
            score += 10

        same_line_latin = _latin_duplicate_from_mixed_line(line)
        if same_line_latin:
            corrected = _correct_from_latin_duplicate(candidates[-1], same_line_latin)
            if corrected != candidates[-1]:
                candidates[-1] = corrected
                score += 4

        if len(candidates) == 3:
            break

    if len(candidates) < 2:
        score -= 30
    return candidates, score


def _owner_label_indexes(lines: list[str]) -> list[int]:
    indexes: list[int] = []
    for index, line in enumerate(lines):
        if _line_has_owner_label(line):
            indexes.append(index)
    return indexes


def _line_has_owner_label(line: str) -> bool:
    lowered = line.lower()
    letters = _letters_only(lowered)
    if _contains(lowered, "\\u0441\\u043e\\u0431\\u0441\\u0442\\u0432\\u0435\\u043d"):
        return True
    if _contains(lowered, "\\u0432\\u043b\\u0430\\u0434\\u0435\\u043b") and len(letters) <= 24:
        return True
    return False


def _line_is_section_boundary(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in _section_markers())


def _looks_like_noise_after_names(line: str) -> bool:
    if _line_is_section_boundary(line):
        return True
    if any(char.isdigit() for char in line):
        return True
    cyrillic_letters = [char for char in line if _is_cyrillic(char)]
    if len(cyrillic_letters) >= 6 and _uppercase_ratio(cyrillic_letters) < 0.55:
        return True
    return False


def _strict_name_candidate_from_line(line: str) -> str | None:
    if _line_has_owner_label(line) or _line_is_section_boundary(line):
        return None

    tokens = _word_tokens(line)
    cyrillic_parts = [
        _titlecase_cyrillic_words(token)
        for token in tokens
        if _is_cyrillic_word(token)
        and _looks_like_name_token(token)
        and _uppercase_ratio([char for char in token if _is_cyrillic(char)]) >= 0.65
    ]
    if not cyrillic_parts:
        return None

    return " ".join(cyrillic_parts[:2])


def _name_candidate_from_line(line: str) -> str | None:
    tokens = _word_tokens(line)
    cyrillic_parts = [
        _titlecase_cyrillic_words(token)
        for token in tokens
        if _is_cyrillic_word(token) and _looks_like_name_token(token)
    ]
    latin_parts = [
        _transliterate_latin_to_cyrillic(token)
        for token in tokens
        if _is_latin_word(token) and len(token) >= 3
    ]
    latin_parts = [part for part in latin_parts if part]

    if cyrillic_parts:
        cyrillic = " ".join(cyrillic_parts)
        if latin_parts:
            return _correct_from_latin_duplicate(cyrillic, " ".join(latin_parts))
        return cyrillic

    return None


def _latin_duplicate_from_line(line: str) -> str | None:
    if _has_cyrillic(line):
        return None

    tokens = [
        token
        for token in _word_tokens(line)
        if _is_latin_word(token) and len(token) >= 3
    ]
    if len(tokens) != 1:
        return None
    return _transliterate_latin_to_cyrillic(tokens[0])


def _latin_duplicate_from_mixed_line(line: str) -> str | None:
    tokens = [
        token
        for token in _word_tokens(line)
        if _is_latin_word(token) and len(token) >= 3
    ]
    if len(tokens) != 1:
        return None
    return _transliterate_latin_to_cyrillic(tokens[0])


def _correct_from_latin_duplicate(cyrillic: str, latin_translit: str | None) -> str:
    if not latin_translit:
        return cyrillic
    if _should_prefer_latin_duplicate(cyrillic, latin_translit):
        return latin_translit
    return cyrillic


def _should_prefer_latin_duplicate(cyrillic: str, latin_translit: str) -> bool:
    cyr_norm = _letters_only(cyrillic).lower()
    lat_norm = _letters_only(latin_translit).lower()
    if not cyr_norm or not lat_norm:
        return False
    if len(lat_norm) < len(cyr_norm):
        return False
    if len(lat_norm) - len(cyr_norm) > 1:
        return False
    return _levenshtein(cyr_norm, lat_norm) <= 2


def _section_markers() -> tuple[str, ...]:
    return (
        _u("\\u0441\\u0443\\u0431\\u044a\\u0435\\u043a\\u0442"),
        _u("\\u0441\\u0443\\u0431"),
        _u("\\u0444\\u0435\\u0434\\u0435\\u0440"),
        _u("\\u0440\\u0435\\u0441\\u043f"),
        _u("\\u043a\\u0440\\u0430\\u0439"),
        _u("\\u043e\\u0431\\u043b"),
        _u("\\u043c\\u043e\\u0441\\u043a\\u0432"),
        _u("\\u0440\\u0430\\u0439\\u043e\\u043d"),
        _u("\\u043f\\u0443\\u043d\\u043a\\u0442"),
        _u("\\u0443\\u043b\\u0438\\u0446"),
        _u("\\u0443\\u043b "),
        _u("\\u0434\\u043e\\u043c"),
        _u("\\u043a\\u0432\\u0430\\u0440\\u0442"),
        _u("\\u043e\\u0441\\u043e\\u0431"),
        _u("\\u043a\\u043e\\u0434"),
        _u("\\u0433\\u0438\\u0431"),
        _u("\\u0434\\u0430\\u0442\\u0430"),
        _u("\\u0432\\u044b\\u0434\\u0430"),
        _u("\\u043d\\u043e\\u0442\\u0430\\u0440"),
        _u("\\u0432\\u043b\\u0430\\u0434\\u0435\\u043b\\u044c\\u0446\\u0430"),
    )


def _word_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in value:
        if char.isalpha() or char == "-":
            current.append(char)
        else:
            if current:
                tokens.append("".join(current).strip("-"))
                current.clear()
    if current:
        tokens.append("".join(current).strip("-"))
    return [token for token in tokens if token]


def _looks_like_name_token(token: str) -> bool:
    lowered = token.lower()
    if len(_letters_only(token)) < 3:
        return False
    return not any(marker in lowered for marker in _section_markers())


def _looks_like_patronymic(value: str) -> bool:
    normalized = _letters_only(value).lower()
    endings = (
        _u("\\u043e\\u0432\\u0438\\u0447"),
        _u("\\u0435\\u0432\\u0438\\u0447"),
        _u("\\u044c\\u0438\\u0447"),
        _u("\\u0438\\u0447"),
        _u("\\u043e\\u0432\\u043d\\u0430"),
        _u("\\u0435\\u0432\\u043d\\u0430"),
        _u("\\u0438\\u0447\\u043d\\u0430"),
        _u("\\u0438\\u043d\\u0438\\u0447\\u043d\\u0430"),
        _u("\\u043e\\u0433\\u043b\\u044b"),
        _u("\\u043a\\u044b\\u0437\\u044b"),
    )
    return any(normalized.endswith(ending) for ending in endings)


def _titlecase_cyrillic_words(text: str) -> str:
    result: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        word = "".join(current)
        if _has_cyrillic(word):
            result.append(word[:1].upper() + word[1:].lower())
        else:
            result.append(word)
        current.clear()

    for char in text:
        if char.isalpha() or char == "-":
            current.append(char)
        else:
            flush()
            result.append(char)
    flush()
    return "".join(result)


def _transliterate_latin_to_cyrillic(value: str) -> str | None:
    value = _letters_only(value).upper()
    if not value:
        return None

    digraphs = {
        "SHCH": "\\u0429",
        "YO": "\\u0401",
        "ZH": "\\u0416",
        "KH": "\\u0425",
        "TS": "\\u0426",
        "CH": "\\u0427",
        "SH": "\\u0428",
        "YU": "\\u042e",
        "YA": "\\u042f",
    }
    chars = {
        "A": "\\u0410",
        "B": "\\u0411",
        "V": "\\u0412",
        "G": "\\u0413",
        "D": "\\u0414",
        "E": "\\u0415",
        "Z": "\\u0417",
        "I": "\\u0418",
        "J": "\\u0419",
        "K": "\\u041a",
        "L": "\\u041b",
        "M": "\\u041c",
        "N": "\\u041d",
        "O": "\\u041e",
        "P": "\\u041f",
        "R": "\\u0420",
        "S": "\\u0421",
        "T": "\\u0422",
        "U": "\\u0423",
        "F": "\\u0424",
        "H": "\\u0425",
        "C": "\\u041a",
        "Y": "\\u0419",
    }

    output: list[str] = []
    index = 0
    while index < len(value):
        matched = False
        for source, target in digraphs.items():
            if value.startswith(source, index):
                output.append(_u(target))
                index += len(source)
                matched = True
                break
        if matched:
            continue
        output.append(_u(chars.get(value[index], "")))
        index += 1

    result = "".join(output)
    return _titlecase_cyrillic_words(result) if result else None


def _is_cyrillic_word(value: str) -> bool:
    letters = _letters_only(value)
    return bool(letters) and all(_is_cyrillic(char) for char in letters)


def _is_latin_word(value: str) -> bool:
    letters = _letters_only(value)
    return bool(letters) and all("A" <= char.upper() <= "Z" for char in letters)


def _has_cyrillic(value: str) -> bool:
    return any(_is_cyrillic(char) for char in value)


def _is_cyrillic(char: str) -> bool:
    return "\u0400" <= char <= "\u04FF"


def _uppercase_ratio(chars: list[str]) -> float:
    if not chars:
        return 0.0
    return sum(char.isupper() for char in chars) / len(chars)


def _letters_only(value: str) -> str:
    return "".join(char for char in value if char.isalpha())


def _contains(value: str, escaped: str) -> bool:
    return _u(escaped) in value


def _u(value: str) -> str:
    return value.encode("ascii").decode("unicode_escape")


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert, delete, replace_cost))
        previous = current
    return previous[-1]
