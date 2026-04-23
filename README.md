# STS FIO MVP

MVP для извлечения `Фамилия`, `Имя`, `Отчество` из изображений СТС.

Пайплайн:

1. OCR: предобработка изображения через OpenCV и распознавание текста через EasyOCR.
2. Дополнительный OCR-кроп блока владельца СТС, чтобы лучше прочитать ФИО.
3. Извлечение сущностей: BERT NER-модель получает OCR-текст и возвращает поля ФИО.
4. Постобработка OCR-артефактов: удаление латинского дубля, исправление кириллической фамилии по транслитерации, отбрасывание низкокачественных латинских фрагментов в отчестве.

CLI в проекте нет. Основной сценарий использования - импортировать функцию в Python-код и вручную передать путь к изображению.

## Установка

Нужен Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Использование в коде

Самый простой пример есть в [example_manual.py](example_manual.py):

```python
from pathlib import Path
from pprint import pprint

from src.sts_fio_mvp import extract_fio_from_image

IMAGE_PATH = Path("data/test3.png")

result = extract_fio_from_image(IMAGE_PATH)

print(result.ocr["text"])
pprint(result.fio)
```

На выходе `result` содержит:

- `result.file`: путь к исходному изображению
- `result.ocr["text"]`: распознанный OCR-текст
- `result.ocr["variant"]`: выбранный вариант предобработки
- `result.fio["surname"]`: фамилия
- `result.fio["name"]`: имя
- `result.fio["patronymic"]`: отчество

Если ФИО на СТС продублировано латиницей, например `КОБАЛАВА / KOBALAVA`, код использует латинский дубль только как подсказку для исправления OCR-ошибки и возвращает кириллицу: `Кобалава`.

## BERT NER

По умолчанию используется `Gherman/bert-base-NER-Russian`. Это token-classification модель для русского NER с person-метками `LAST_NAME`, `FIRST_NAME`, `MIDDLE_NAME`, поэтому она лучше подходит для СТС, чем QA.

```python
from src.sts_fio_mvp import extract_fio_from_image

result = extract_fio_from_image("data/test3.png")
```

Для более качественного production-варианта стоит дообучить BERT на OCR-текстах СТС с метками:

- `LAST_NAME` или `SURNAME`
- `FIRST_NAME` или `NAME`
- `MIDDLE_NAME` или `PATRONYMIC`

И затем использовать его так:

```python
from src.sts_fio_mvp import ExtractorMode, extract_fio_from_image

result = extract_fio_from_image(
    "data/test3.png",
    extractor_mode=ExtractorMode.TOKEN_CLASSIFICATION,
    model_name="path/to/fio-bert",
)
```

BIO/BIOLU-префиксы вида `B-LAST_NAME`, `I-LAST_NAME`, `U-FIRST_NAME` поддерживаются через HuggingFace aggregation pipeline.

QA-режим оставлен только как экспериментальный fallback:

```python
from src.sts_fio_mvp import ExtractorMode, extract_fio_from_image

result = extract_fio_from_image(
    "data/test3.png",
    extractor_mode=ExtractorMode.QA,
)
```

## GPU

В машине должен быть NVIDIA GPU и PyTorch с CUDA. Проверка:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Если `torch.cuda.is_available()` возвращает `True`, запускай так:

```python
result = extract_fio_from_image(
    "data/test3.png",
    use_gpu=True,
    bert_device=0,
    progress=print,
)
```

Если возвращает `False`, установлен CPU-only PyTorch. Для Windows нужно поставить CUDA-сборку PyTorch из официального селектора: https://pytorch.org/get-started/locally/

## Заметки

- EasyOCR скачивает OCR-модели при первом запуске.
- HuggingFace-модель тоже скачивается при первом запуске, если `model_name` не указывает на локальную директорию.
- `use_gpu=False` отключает GPU для OCR.
- `bert_device=-1` запускает BERT на CPU, `bert_device=0` - на первой CUDA GPU.
