from pathlib import Path
from pprint import pprint
import sys


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch

from src.sts_fio_mvp import extract_fio_from_image


IMAGE_PATH = Path("data/test5.jpg")
USE_GPU = torch.cuda.is_available()

print(f"PyTorch CUDA available: {USE_GPU}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

result = extract_fio_from_image(
    IMAGE_PATH,
    debug_dir="results/debug",
    use_gpu=USE_GPU,
    bert_device=0 if USE_GPU else -1,
    progress=lambda message: print(f"[sts-fio] {message}", flush=True),
)

print("OCR text:")
print(result.ocr["text"])

print("\nFIO:")
pprint(result.fio)
