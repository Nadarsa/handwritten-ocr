from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Type

from PIL import Image


ImageLike = Union[str, Image.Image]


class BaseOCRModel(ABC):
    """Base class for OCR/HTR model inference."""

    @abstractmethod
    def load(self, device: str | None = None) -> None:
        """Load model weights and move to the specified device."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: ImageLike) -> str:
        """Run inference on an image and return recognized text."""
        raise NotImplementedError


def _models_registry() -> Dict[str, str]:
    """Internal mapping from short model name to full import path."""
    return {
        "trocr_kazars": "models.ocr_models.trocr:TrOCRKazars",
        "trocr_raxtemur": "models.ocr_models.trocr:TrOCRRaxtemur",
        "trocr_cyrillic": "models.ocr_models.trocr:TrOCRCyrillic",
        # Kansallisarkisto/cyrillic-htr-model имеет нестандартную структуру репозитория
        # (процессор в подпапке `processor/`), из-за чего автоматическая загрузка через
        # transformers в текущем окружении не работает. Обёртка для него есть в
        # ocr_models.trocr.TrOCRKansallis, но по умолчанию мы не регистрируем её
        # в фабрике, чтобы не ломать код. Для использования этой модели
        # потребуется ручная настройка и локальное скачивание процессора.
        # "trocr_kansallis": "models.ocr_models.trocr:TrOCRKansallis",
        # Обёртки на базе модели Peter/ReadingPipeline показали очень низкое
        # качество на целевом датасете (современные рукописные тетради HWR200),
        # поэтому мы исключили их из фабрики get_model. Код оставлен в модулях
        # models.ocr_models.peter_crnn и models.ocr_models.reading_pipeline_peter
        # только для экспериментов; в основном проекте они не используются.
        # "peter_crnn": "models.ocr_models.peter_crnn:PeterCRNN",
        # ReadingPipeline-Peter (ai-forever) — полноценный пайплайн сегментации+OCR
        # для исторических рукописей Петра I. Чтобы он заработал, нужно:
        # - установить репозиторий ai-forever/ReadingPipeline (пакет ocrpipeline),
        # - скачать содержимое ai-forever/ReadingPipeline-Peter в папку
        #   `reading_pipeline_peter/` рядом с проектом.
        # "reading_pipeline_peter": "models.ocr_models.reading_pipeline_peter:ReadingPipelinePeter",
    }


def get_model(name: str) -> BaseOCRModel:
    """
    Factory to create and load an OCR model by its short name.

    Example:
        from models.ocr_models import get_model
        model = get_model("trocr_kazars")
        text = model.predict("path/to/image.jpg")
    """
    registry = _models_registry()
    if name not in registry:
        raise ValueError(f"Unknown OCR model: {name!r}. Available: {sorted(registry.keys())}")

    spec = registry[name]
    module_path, class_name = spec.split(":", 1)

    import importlib

    module = importlib.import_module(module_path)
    model_class: Type[BaseOCRModel] = getattr(module, class_name)

    model = model_class()
    model.load()
    return model
