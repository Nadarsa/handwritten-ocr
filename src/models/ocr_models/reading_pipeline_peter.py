"""
ВНИМАНИЕ: обёртка ReadingPipelinePeter оставлена только для экспериментов.
Из-за тяжёлых зависимостей и слабого качества на нашем целевом датасете
(современные рукописные тетради HWR200) модель исключена из фабрики
`models.ocr_models.get_model` и не используется в основном проекте.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from . import BaseOCRModel, ImageLike


class ReadingPipelinePeter(BaseOCRModel):
    """
    Обёртка вокруг Peter-части ReadingPipeline:
    - Segmentation + OCR для исторических рукописей Петра I.

    Зависит от внешнего пакета `ReadingPipeline` (код в репозитории
    ai-forever/ReadingPipeline) и весов с HuggingFace:
    - ai-forever/ReadingPipeline-Peter
    """

    def __init__(self) -> None:
        self.pipeline = None

    def load(self, device: str | None = None) -> None:
        """
        Ожидается, что пользователь сам:
        - клонировал `ai-forever/ReadingPipeline` и установил его как пакет,
        - скачал содержимое репозитория `ai-forever/ReadingPipeline-Peter`
          в локальную папку (например `reading_pipeline_peter/`),
        - указал путь к `pipeline_config.json`, где прописаны реальные
          `model_path` и `config_path` для segm/ocr.

        Здесь мы только создаём PipelinePredictor по указанному пути.
        """
        try:
            from ocrpipeline.predictor import PipelinePredictor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ReadingPipeline не установлен. "
                "Установи/подключи репозиторий ai-forever/ReadingPipeline "
                "как пакет (ocrpipeline) перед использованием "
                "ReadingPipelinePeter."
            ) from exc

        config_path = "reading_pipeline_peter/pipeline_config.json"
        self.pipeline = PipelinePredictor(config_path)

    def predict(self, image: ImageLike) -> str:
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            # PIL.Image -> BGR
            img_rgb = np.array(image.convert("RGB"))
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        _, pred = self.pipeline(img)

        lines: List[tuple[int, str]] = []
        for p in pred.get("predictions", []):
            text = p.get("text", "")
            if not text:
                continue
            line_idx = int(p.get("line_idx", 0))
            col_idx = int(p.get("column_idx", 0))
            word_idx = int(p.get("word_idx", 0))
            # тройка индексов для устойчивой сортировки
            lines.append(((line_idx, col_idx, word_idx), text))

        # сортируем сначала по строкам, затем по колонкам и позиции слова
        lines.sort(key=lambda x: x[0])

        # группируем по line_idx для формирования текста по строкам
        result_lines: List[str] = []
        current_line = None
        current_words: List[str] = []

        for (line_idx, col_idx, word_idx), text in lines:
            if current_line is None:
                current_line = line_idx
            if line_idx != current_line:
                if current_words:
                    result_lines.append(" ".join(current_words))
                current_words = [text]
                current_line = line_idx
            else:
                current_words.append(text)

        if current_words:
            result_lines.append(" ".join(current_words))

        return "\n".join(result_lines)
