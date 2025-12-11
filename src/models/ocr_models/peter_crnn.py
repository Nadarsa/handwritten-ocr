"""
ВНИМАНИЕ: обёртка PeterCRNN оставлена только для экспериментов.
По результатам тестов на нашем датасете (страницы HWR200) качество
оказалось существенно хуже, чем у TrOCR‑моделей, поэтому модель
исключена из фабрики `models.ocr_models.get_model` и не используется
в основном проекте.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from . import BaseOCRModel, ImageLike


class _GlobalMaxPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-2, keepdim=True)[0]


def _get_resnet34_backbone() -> nn.Module:
    # Предобученность не критична, так как веса полностью переопределяются
    from torchvision.models import resnet34

    m = resnet34(pretrained=False)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    blocks = [input_conv, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class _BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class _CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        time_feature_count: int = 256,
        lstm_hidden: int = 256,
        lstm_len: int = 3,
    ) -> None:
        super().__init__()
        self.feature_extractor = _get_resnet34_backbone()
        self.global_maxpool = _GlobalMaxPool2d()
        self.bilstm = _BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, time_feature_count),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_feature_count, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.global_maxpool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x = self.bilstm(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=2).permute(1, 0, 2)
        return x


class _RescalePaddingImage:
    def __init__(self, output_height: int, output_width: int) -> None:
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        new_width = int(w * (self.output_height / h))
        new_width = min(new_width, self.output_width)
        image = cv2.resize(image, (new_width, self.output_height), interpolation=cv2.INTER_LINEAR)
        if new_width < self.output_width:
            image = np.pad(
                image,
                ((0, 0), (0, self.output_width - new_width), (0, 0)),
                "constant",
                constant_values=0,
            )
        return image


class _Normalize:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0


class _MoveChannels:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.moveaxis(image, -1, 0)


class _InferenceTransform:
    def __init__(self, height: int, width: int) -> None:
        self.rescale = _RescalePaddingImage(height, width)
        self.normalize = _Normalize()
        self.move_channels = _MoveChannels()

    def __call__(self, images: List[np.ndarray]) -> torch.Tensor:
        transformed = []
        for img in images:
            x = self.rescale(img)
            x = self.normalize(x)
            x = self.move_channels(x)
            transformed.append(x)
        arr = np.stack(transformed, 0)
        return torch.from_numpy(arr)


class _Tokenizer:
    def __init__(self, alphabet: str) -> None:
        # Добавляем специальные токены CTC: blank и OOV
        self.blank_id = 0
        self.oov_id = 1
        self.char_map = {ch: idx + 2 for idx, ch in enumerate(alphabet)}
        self.char_map["<BLANK>"] = self.blank_id
        self.char_map["<OOV>"] = self.oov_id
        self.rev_char_map = {v: k for k, v in self.char_map.items()}

    def decode_best_path(self, logits: torch.Tensor) -> List[str]:
        # logits: (T, B, C)
        pred = torch.argmax(logits.detach().cpu(), dim=-1).permute(1, 0).numpy()  # (B, T)
        texts: List[str] = []
        for seq in pred:
            chars: List[str] = []
            prev = None
            for token in seq:
                if token in (self.blank_id, self.oov_id):
                    prev = token
                    continue
                if prev is not None and token == prev:
                    prev = token
                    continue
                chars.append(self.rev_char_map.get(int(token), ""))
                prev = token
            texts.append("".join(chars))
        return texts


@dataclass
class _PeterConfig:
    alphabet: str
    image_height: int
    image_width: int


def _load_peter_config(path: str) -> _PeterConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    img = cfg.get("image", {})
    return _PeterConfig(
        alphabet=cfg["alphabet"],
        image_height=int(img.get("height", 128)),
        image_width=int(img.get("width", 1024)),
    )


class PeterCRNN(BaseOCRModel):
    """
    CRNN-модель из OCR-model, дообученная на Peter dataset
    (веса взяты из ai-forever/ReadingPipeline-Peter).

    Для полноценных страниц используется детектор PaddleOCR для нарезки
    на строки/области.
    """

    def __init__(self) -> None:
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[_CRNN] = None
        self.transform: Optional[_InferenceTransform] = None
        self.tokenizer: Optional[_Tokenizer] = None
        self.detector = None

    def load(self, device: str | None = None) -> None:
        if device is not None:
            self.device = device

        cfg = _load_peter_config("reading_pipeline_peter/ocr/ocr_config.json")
        self.transform = _InferenceTransform(cfg.image_height, cfg.image_width)
        self.tokenizer = _Tokenizer(cfg.alphabet)

        num_classes = len(self.tokenizer.char_map)
        model = _CRNN(num_classes=num_classes)
        state_dict = torch.load(
            "reading_pipeline_peter/ocr/ocr_model.ckpt",
            map_location=self.device,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        # Детектор строк — PaddleOCR (как в TrOCR-пайплайне).
        # Если он не поднимается, это критическая ошибка, а не тихий фолбэк.
        from paddleocr import PaddleOCR

        self.detector = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def _crop_lines(self, image: Image.Image) -> List[Image.Image]:
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
        if self.model is None or self.transform is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        crops = self._crop_lines(image)

        texts: List[str] = []
        for crop in crops:
            # Pillow -> BGR numpy, как ожидает transform
            arr = np.array(crop)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            batch = self.transform([arr]).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)  # (T, B, C)

            preds = self.tokenizer.decode_best_path(logits)
            text = preds[0].strip()
            if text:
                texts.append(text)

        return "\n".join(texts)
