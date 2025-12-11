from __future__ import annotations

from typing import List

from .ocr_models import BaseOCRModel, ImageLike, get_model as _get_model


AVAILABLE_OCR_MODELS: List[str] = [
    "trocr_kazars",
    "trocr_raxtemur",
    "trocr_cyrillic",
    # Обёртки на базе Peter/ReadingPipeline (`peter_crnn`, `reading_pipeline_peter`)
    # отключены из-за очень низкого качества на целевом датасете (страницы HWR200)
    # и не включаются в список моделей по умолчанию.
]


def get_ocr_model(name: str) -> BaseOCRModel:
    """
    Вернуть загруженную OCR-модель по короткому имени.

    Пример:
        from models import get_ocr_model
        model = get_ocr_model(\"trocr_kazars\")
        text = model.predict(\"path/to/image.jpg\")
    """
    return _get_model(name)
