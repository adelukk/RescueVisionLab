
import json
import shutil
from pathlib import Path
from collections import Counter

# Папка нашего проекта
ROOT = Path(r"C:\RescueVisionLab")

# Исходные изображения, разметка и разбиение
RAW = ROOT / "datasets" / "raw" / "pdd"
IMAGES = RAW / "images"
LABELS = RAW / "labels"
SPLIT_FILE = RAW / "split.json"

# Папка для обучения YOLO
OUTPUT = ROOT / "datasets" / "processed" / "yolo"

# Читаем готовое разбиение
with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

records = manifest["records"]

if len(records) != 879:
    raise ValueError(
        f"Ожидалось 879 записей, найдено {len(records)}"
    )

# Сначала проверяем все файлы
seen = set()
counts = Counter()

for record in records:
    filename = record["file"]
    split = record["split"]

    if split not in ("train", "val", "test"):
        raise ValueError(f"Неизвестная выборка: {split}")

    if filename in seen:
        raise ValueError(f"Повтор файла: {filename}")

    seen.add(filename)

    image = IMAGES / filename
    label = LABELS / (Path(filename).stem + ".txt")

    if not image.is_file():
        raise FileNotFoundError(f"Нет изображения: {image}")

    if not label.is_file():
        raise FileNotFoundError(f"Нет разметки: {label}")

    # Проверяем формат YOLO:
    # class x_center y_center width height
    for line in label.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(f"Ошибка разметки: {label}")

        if parts[0] != "0":
            raise ValueError(f"Неожиданный класс: {label}")

        values = [float(x) for x in parts[1:]]

        if not all(0 <= x <= 1 for x in values):
            raise ValueError(f"Неверные координаты: {label}")

        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Нулевой размер объекта: {label}")

    counts[split] += 1

print("Проверка пройдена.")
print("Распределение:", dict(counts))

# Создаём папки
for split in ("train", "val", "test"):
    (OUTPUT / "images" / split).mkdir(
        parents=True, exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True, exist_ok=True
    )

# Копируем файлы по split.json
for record in records:
    filename = record["file"]
    split = record["split"]
    label_name = Path(filename).stem + ".txt"

    shutil.copy2(
        IMAGES / filename,
        OUTPUT / "images" / split / filename
    )

    shutil.copy2(
        LABELS / label_name,
        OUTPUT / "labels" / split / label_name
    )

print("\nГОТОВО! Датасет подготовлен.")

for split in ("train", "val", "test"):
    n_images = len(list(
        (OUTPUT / "images" / split).iterdir()
    ))
    n_labels = len(list(
        (OUTPUT / "labels" / split).iterdir()
    ))

    print(
        f"{split}: "
        f"{n_images} изображений, "
        f"{n_labels} разметок"
    )
