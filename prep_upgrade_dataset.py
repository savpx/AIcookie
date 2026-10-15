"""Подготовка датасета для детекции отдельных строк улучшений (upgrade1..upgrade6).

Берёт размеченный скриншот из X-AnyLabeling (JSON + PNG), конвертирует
прямоугольники в YOLO-формат, генерирует аугментации (UI-safe: без флипов)
и раскладывает результат в images/labels train/val. Старые данные класса
"upgrade" (единый бокс панели) удаляются, т.к. заменяются новыми.
"""
import json
import os
import shutil
import random

import cv2
import numpy as np
import albumentations as A

SRC_DIR = "C:/Users/zloym/OneDrive/\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f/Screenshots"
SRC_IMG = os.path.join(
    SRC_DIR, "\u0421\u043d\u0438\u043c\u043e\u043a \u044d\u043a\u0440\u0430\u043d\u0430 2026-10-15 112327.png")
SRC_JSON = SRC_IMG[:-4] + ".json"

# Порядок классов строго фиксирован (индексы совпадают с data.yaml)
CLASS_NAMES = ["game", "close", "bonus",
               "upgrade1", "upgrade2", "upgrade3",
               "upgrade4", "upgrade5", "upgrade6"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

N_TRAIN_AUG = 40
N_VAL_AUG = 8
SEED = 1234


def load_boxes(json_path):
    """JSON X-AnyLabeling -> список (cx, cy, w, h, class_idx) в YOLO-формате."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    img_w = data["imageWidth"]
    img_h = data["imageHeight"]
    boxes = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_TO_IDX:
            continue
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        boxes.append((cx, cy, w, h, CLASS_TO_IDX[label]))
    return img_w, img_h, boxes


def build_transform():
    """UI-safe аугментации: без горизонтальных/вертикальных флипов и поворотов."""
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.12, rotate_limit=4,
                border_mode=cv2.BORDER_REPLICATE, p=0.9,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20,
                p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"],
            min_visibility=0.3),
    )


def write_label(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for cx, cy, w, h, cls in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def remove_old_upgrade_files():
    removed = 0
    for split in ("train", "val"):
        for folder in (f"images/{split}", f"labels/{split}"):
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if name.startswith("cookie_upgrade_"):
                    os.remove(os.path.join(folder, name))
                    removed += 1
    print(f"Удалено старых файлов upgrade: {removed}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    img_w, img_h, boxes = load_boxes(SRC_JSON)
    print(f"Изображение {img_w}x{img_h}, размечено боксов: {len(boxes)}")
    for cx, cy, w, h, cls in boxes:
        print(f"  {CLASS_NAMES[cls]}: cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")

    # cv2.imread не читает пути с кириллицей -> читаем через numpy
    buf = np.fromfile(SRC_IMG, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {SRC_IMG}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for split in ("train", "val"):
        os.makedirs(f"images/{split}", exist_ok=True)
        os.makedirs(f"labels/{split}", exist_ok=True)

    transform = build_transform()

    def save_augmented(split, index, total):
        bboxes = [(b[0], b[1], b[2], b[3]) for b in boxes]
        labels = [b[4] for b in boxes]
        try:
            res = transform(image=image, bboxes=bboxes, class_labels=labels)
        except Exception as exc:  # noqa: BLE001
            print(f"  пропуск аугментации: {exc}")
            return False
        aug_img = res["image"]
        aug_boxes = [
            (b[0], b[1], b[2], b[3], int(l))
            for b, l in zip(res["bboxes"], res["class_labels"])
        ]
        stem = f"upgrade_rows_{split}_{index:04d}"
        cv2.imwrite(f"images/{split}/{stem}.png",
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        write_label(f"labels/{split}/{stem}.txt", aug_boxes)
        return True

    # Оригинал -> train (без искажений)
    cv2.imwrite("images/train/upgrade_rows_train_orig.png",
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_label("labels/train/upgrade_rows_train_orig.txt", boxes)

    made_train = 0
    for i in range(N_TRAIN_AUG):
        if save_augmented("train", i, N_TRAIN_AUG):
            made_train += 1

    made_val = 0
    for i in range(N_VAL_AUG):
        if save_augmented("val", i, N_VAL_AUG):
            made_val += 1

    print(f"Сгенерировано: train={made_train + 1}, val={made_val}")

    remove_old_upgrade_files()
    print("Готово.")


if __name__ == "__main__":
    main()
