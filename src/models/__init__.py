from .inference import get_ocr_model, AVAILABLE_OCR_MODELS
from .ocr_models import BaseOCRModel as OCRBaseModel, ImageLike

__all__ = ["OCRBaseModel", "ImageLike", "get_ocr_model", "AVAILABLE_OCR_MODELS"]
