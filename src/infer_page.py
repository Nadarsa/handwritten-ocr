from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from models import AVAILABLE_OCR_MODELS, get_ocr_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Инференс рукописного OCR по одной странице (полное изображение)."
    )
    parser.add_argument("model", choices=AVAILABLE_OCR_MODELS, help="Имя модели из списка AVAILABLE_OCR_MODELS")
    parser.add_argument(
        "image",
        type=str,
        help="Путь к изображению или директории с изображениями",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Путь к txt-файлу (для одного изображения) или к директории (для папки). "
            "По умолчанию txt-файлы создаются рядом с исходными изображениями."
        ),
    )
    args = parser.parse_args()

    model = get_ocr_model(args.model)
    image_path = Path(args.image)

    def save_result(img_path: Path, text: str, out_dir: Path | None = None) -> Path:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{img_path.stem}.{args.model}.txt"
        else:
            out_path = img_path.with_suffix(f".{args.model}.txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        return out_path

    if image_path.is_dir():
        exts: List[str] = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images: List[Path] = []
        for pattern in exts:
            images.extend(sorted(image_path.rglob(pattern)))

        if not images:
            print(f"В директории {image_path} не найдено изображений с расширениями jpg/jpeg/png.")
            return

        out_dir = Path(args.output) if args.output is not None else None
        print(f"Модель: {args.model}")
        print(f"Всего изображений: {len(images)}")
        for img in images:
            text = model.predict(str(img))
            out_path = save_result(img, text, out_dir=out_dir)
            print(f"{img} -> {out_path}")
    else:
        if args.output is None:
            out_path = image_path.with_suffix(f".{args.model}.txt")
        else:
            out_path = Path(args.output)

        text = model.predict(str(image_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        print(f"Модель: {args.model}")
        print(f"Изображение: {image_path}")
        print(f"Результат сохранён в: {out_path}")


if __name__ == "__main__":
    main()
