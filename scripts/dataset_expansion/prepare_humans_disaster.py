import json
import shutil
import zipfile
from pathlib import Path


ZIP_PATH = Path(
    r"C:\Users\Аделина\Downloads\humans in disaster.v1i.yolov8.zip"
)

ROOT = Path(
    r"datasets\raw\combined_sources\humans_disaster"
)

IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
METADATA_DIR = ROOT / "metadata"

PERSON_CLASS = 1
SEED = 42

SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label(text, source_name):
    boxes = []

    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid label format: {source_name}:{line_number}"
            )

        cls = int(values[0])
        xc, yc, bw, bh = map(float, values[1:])

        if not all(0 <= value <= 1 for value in (xc, yc, bw, bh)):
            raise ValueError(
                f"Coordinates out of range: {source_name}:{line_number}"
            )

        # Humans in Disaster:
        # 0 = car
        # 1 = person
        # 2 = truck
        if cls != PERSON_CLASS:
            continue

        if bw <= 0 or bh <= 0:
            continue

        # Convert person class 1 -> our class 0.
        boxes.append((0, xc, yc, bw, bh))

    return boxes


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(ZIP_PATH)

    if ROOT.exists():
        shutil.rmtree(ROOT)

    IMAGES_DIR.mkdir(parents=True)
    LABELS_DIR.mkdir(parents=True)
    METADATA_DIR.mkdir(parents=True)

    records = []

    source_images = 0
    source_person_boxes = 0
    kept_images = 0
    removed_no_person = 0

    split_stats = {}

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())

        for source_split in SOURCE_SPLITS:
            image_prefix = f"{source_split}/images/"

            image_names = sorted(
                name
                for name in names
                if name.startswith(image_prefix)
                and Path(name).suffix.lower() in IMAGE_EXTENSIONS
            )

            stats = {
                "source_images": 0,
                "kept_images": 0,
                "person_boxes": 0,
                "removed_no_person": 0,
            }

            for image_name in image_names:
                stats["source_images"] += 1
                source_images += 1

                image_path_in_zip = Path(image_name)
                stem = image_path_in_zip.stem

                label_name = (
                    f"{source_split}/labels/{stem}.txt"
                )

                if label_name not in names:
                    raise FileNotFoundError(
                        f"Missing label for {image_name}"
                    )

                label_text = archive.read(label_name).decode(
                    "utf-8-sig"
                )

                boxes = parse_label(label_text, label_name)

                if not boxes:
                    stats["removed_no_person"] += 1
                    removed_no_person += 1
                    continue

                # Prefix filenames so collisions between original
                # Roboflow splits are impossible.
                uid = f"{source_split}_{stem}"

                extension = image_path_in_zip.suffix.lower()

                output_image = IMAGES_DIR / f"{uid}{extension}"
                output_label = LABELS_DIR / f"{uid}.txt"

                output_image.write_bytes(
                    archive.read(image_name)
                )

                output_label.write_text(
                    "\n".join(
                        f"{cls} "
                        f"{xc:.8f} {yc:.8f} "
                        f"{bw:.8f} {bh:.8f}"
                        for cls, xc, yc, bw, bh in boxes
                    )
                    + "\n",
                    encoding="utf-8",
                )

                record = {
                    "uid": uid,
                    "source_split": source_split,
                    "image": output_image.as_posix(),
                    "label": output_label.as_posix(),
                    "boxes": len(boxes),
                }

                records.append(record)

                stats["kept_images"] += 1
                stats["person_boxes"] += len(boxes)

                kept_images += 1
                source_person_boxes += len(boxes)

            split_stats[source_split] = stats

    # We intentionally pool the original Roboflow train/valid/test.
    # Our project will create its own deterministic train/val split.
    #
    # The only final project test set remains the untouched PDD test.
    import random

    rng = random.Random(SEED)
    rng.shuffle(records)

    val_count = round(len(records) * 0.10)

    val_records = records[:val_count]
    train_records = records[val_count:]

    train_records = sorted(
        train_records,
        key=lambda item: item["uid"],
    )
    val_records = sorted(
        val_records,
        key=lambda item: item["uid"],
    )

    train_ids = {item["uid"] for item in train_records}
    val_ids = {item["uid"] for item in val_records}

    overlap = train_ids & val_ids

    if overlap:
        raise RuntimeError(
            f"Train/val overlap: {len(overlap)}"
        )

    train_boxes = sum(
        item["boxes"] for item in train_records
    )
    val_boxes = sum(
        item["boxes"] for item in val_records
    )

    manifest = {
        "dataset": "Humans in Disaster",
        "source": "Roboflow Universe",
        "version": 1,
        "license": "CC BY 4.0",
        "seed": SEED,
        "class_mapping": {
            "source_person_class": 1,
            "target_person_class": 0,
        },
        "source_statistics": split_stats,
        "statistics": {
            "source_images": source_images,
            "kept_images": kept_images,
            "removed_no_person": removed_no_person,
            "person_boxes": source_person_boxes,
            "train": {
                "images": len(train_records),
                "boxes": train_boxes,
            },
            "val": {
                "images": len(val_records),
                "boxes": val_boxes,
            },
        },
        "train": train_records,
        "val": val_records,
    }

    manifest_path = (
        METADATA_DIR / "humans_disaster_split.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 42)
    print(" Humans in Disaster preparation")
    print("=" * 42)

    for split in SOURCE_SPLITS:
        stats = split_stats[split]

        print(f"\nSOURCE {split.upper()}")
        print("  Images:", stats["source_images"])
        print("  Kept:", stats["kept_images"])
        print("  Removed without person:", stats["removed_no_person"])
        print("  Person boxes:", stats["person_boxes"])

    print("\n" + "=" * 42)
    print("FINAL")
    print("=" * 42)

    print("Source images:", source_images)
    print("Kept images:", kept_images)
    print("Removed without person:", removed_no_person)
    print("Person boxes:", source_person_boxes)

    print("\nTRAIN")
    print("  Images:", len(train_records))
    print("  Boxes:", train_boxes)

    print("\nVAL")
    print("  Images:", len(val_records))
    print("  Boxes:", val_boxes)

    print("\nTrain/val overlap:", len(overlap))

    print("\nSaved:")
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()