from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from models import AVAILABLE_OCR_MODELS, get_ocr_model
from metrics import cer, wer, batch_cer_wer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка OCR-модели по папке: изображения + эталонные .txt с теми же именами."
    )
    parser.add_argument("model", choices=AVAILABLE_OCR_MODELS, help="Имя модели из списка AVAILABLE_OCR_MODELS")
    parser.add_argument("images", type=str, help="Путь к директории с изображениями")
    parser.add_argument(
        "--exts",
        type=str,
        default="jpg,jpeg,png,JPG,JPEG,PNG",
        help="Список расширений изображений через запятую (по умолчанию: jpg,jpeg,png,...)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не печатать построчные ошибки, только итоговые метрики.",
    )
    args = parser.parse_args()

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        raise SystemExit(f"{img_dir} не является директорией")

    exts = [f"*.{e.strip()}" for e in args.exts.split(",") if e.strip()]
    images: List[Path] = []
    for pattern in exts:
        images.extend(sorted(img_dir.rglob(pattern)))

    if not images:
        raise SystemExit(f"В директории {img_dir} не найдено изображений с указанными расширениями")

    model = get_ocr_model(args.model)
    pairs: List[Tuple[str, str]] = []

    for img_path in images:
        gt_path = img_path.with_suffix(".txt")
        if not gt_path.is_file():
            if not args.quiet:
                print(f"[WARN] Нет эталона для {img_path} (ожидался {gt_path}), пропускаю")
            continue

        gt_text = gt_path.read_text(encoding="utf-8")
        pred_text = model.predict(str(img_path))

        c = cer(gt_text, pred_text)
        w = wer(gt_text, pred_text)
        pairs.append((gt_text, pred_text))

        if not args.quiet:
            print(f"{img_path.name}: CER={c:.3f}, WER={w:.3f}")

    if not pairs:
        raise SystemExit("Нет ни одной пары (изображение + .txt), по которым можно считать метрики")

    avg_cer, avg_wer = batch_cer_wer(pairs)
    print(f"Модель: {args.model}")
    print(f"Всего примеров: {len(pairs)}")
    print(f"Средний CER: {avg_cer:.3f}")
    print(f"Средний WER: {avg_wer:.3f}")


if __name__ == "__main__":
    main()
