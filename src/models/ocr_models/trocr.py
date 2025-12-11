from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from PIL import Image
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import BaseOCRModel, ImageLike


@dataclass
class _TrOCRConfig:
    model_name: str
    processor_name: Optional[str] = None


class _BaseTrOCRModel(BaseOCRModel):
    """Shared minimalistic wrapper around TrOCR models."""

    CONFIG: _TrOCRConfig

    def __init__(self) -> None:
        self.processor: Optional[TrOCRProcessor] = None
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = None  # PaddleOCR detector for full-page text

    def load(self, device: str | None = None) -> None:
        if device is not None:
            self.device = device

        processor_name = self.CONFIG.processor_name or self.CONFIG.model_name
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.CONFIG.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Инициализация детектора строк/областей текста для работы с целой страницей.
        # Если PaddleOCR не ставится/не инициализируется, это критическая ошибка:
        # без него модель увидит только одну «строку» (всю страницу), что ломает ТЗ.
        from paddleocr import PaddleOCR  # пусть ImportError поднимется наружу

        self.detector = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def _crop_lines(self, image: Image.Image) -> List[Image.Image]:
        """Разбивает страницу на кропы строк/областей с помощью PaddleOCR.

        Если детектор недоступен или не вернул полигоны, возвращает исходное изображение как единственный кроп.
        """
        if self.detector is None:
            raise RuntimeError("PaddleOCR detector is not initialized. Call load() successfully before predict().")

        img_array = np.array(image)
        result = self.detector.predict(img_array)

        if not result or len(result) == 0:
            return [image]

        res = result[0]
        polys = res.get("dt_polys")
        if not polys:
            return [image]

        # Сортируем полигоны сверху-вниз, слева-направо (приблизительный порядок чтения).
        sorted_polys = sorted(
            polys,
            key=lambda p: (np.mean([pt[1] for pt in p]), np.mean([pt[0] for pt in p])),
        )

        crops: List[Image.Image] = []
        for poly in sorted_polys:
            pts = np.array(poly)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            padding = 5
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(image.width, int(x_max) + padding)
            y_max = min(image.height, int(y_max) + padding)

            crop = image.crop((x_min, y_min, x_max, y_max))
            crops.append(crop)

        return crops or [image]

    def predict(self, image: ImageLike) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        # Для целой страницы сначала детектируем строки/области, затем прогоняем каждый кроп через TrOCR.
        crops = self._crop_lines(image)

        texts: List[str] = []
        for crop in crops:
            pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = text.strip()
            if text:
                texts.append(text)

        # Собираем полный текст страницы построчно.
        return "\n".join(texts)


class TrOCRKazars(_BaseTrOCRModel):
    """kazars24/trocr-base-handwritten-ru — современная рукопись."""

    CONFIG = _TrOCRConfig(
        model_name="kazars24/trocr-base-handwritten-ru",
    )


class TrOCRRaxtemur(_BaseTrOCRModel):
    """raxtemur/trocr-base-ru — TrOCR для русского, обучен на синтетике StackMix."""

    CONFIG = _TrOCRConfig(
        model_name="raxtemur/trocr-base-ru",
    )


class TrOCRCyrillic(_BaseTrOCRModel):
    """cyrillic-trocr/trocr-handwritten-cyrillic — рукописная историческая/церковнославянская кириллица."""

    CONFIG = _TrOCRConfig(
        model_name="cyrillic-trocr/trocr-handwritten-cyrillic",
    )


class TrOCRKansallis(_BaseTrOCRModel):
    """Kansallisarkisto/cyrillic-htr-model — TrOCR-large для исторических рукописей."""

    CONFIG = _TrOCRConfig(
        model_name="Kansallisarkisto/cyrillic-htr-model",
        processor_name=None,  # загрузка по умолчанию не работает из-за структуры репозитория
    )
