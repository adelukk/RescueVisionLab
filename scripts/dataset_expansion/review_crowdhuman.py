from pathlib import Path
import json
import random

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
METADATA_DIR = DATASET_ROOT / "metadata"

MANIFEST_FILE = (
    METADATA_DIR
    / "selected_images.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "crowdhuman_visual_check"
)

RANDOM_SEED = 42

REVIEW_TARGETS = {
    "2-5": 6,
    "6-10": 6,
    "11-20": 6,
    ">20": 6,
}


def read_yolo_labels(label_path):
    boxes = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Invalid label: {label_path}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise RuntimeError(
                f"Unexpected class: {class_id}"
            )

        xc, yc, bw, bh = map(
            float,
            parts[1:],
        )

        boxes.append(
            (xc, yc, bw, bh)
        )

    return boxes


def yolo_to_xyxy(
    box,
    width,
    height,
):
    xc, yc, bw, bh = box

    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def draw_image(
    image_path,
    label_path,
    output_path,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    boxes = read_yolo_labels(
        label_path
    )

    line_width = max(
        2,
        round(
            min(width, height) / 250
        ),
    )

    for number, box in enumerate(
        boxes,
        start=1,
    ):
        x1, y1, x2, y2 = (
            yolo_to_xyxy(
                box,
                width,
                height,
            )
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=line_width,
        )

        draw.text(
            (
                x1 + 2,
                max(0, y1 - 14),
            ),
            f"PERSON {number}",
            fill="lime",
        )

    image.save(
        output_path,
        quality=92,
    )

    return len(boxes)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove previous visual review.
    for path in OUTPUT_DIR.glob(
        "*.jpg"
    ):
        path.unlink()

    data = json.loads(
        MANIFEST_FILE.read_text(
            encoding="utf-8"
        )
    )

    buckets = {
        category: []
        for category in REVIEW_TARGETS
    }

    for record in data["records"]:
        category = record["category"]

        if category not in buckets:
            continue

        image_id = record["image_id"]

        image_path = (
            IMAGES_DIR
            / f"{image_id}.jpg"
        )

        label_path = (
            LABELS_DIR
            / f"{image_id}.txt"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        if not label_path.exists():
            raise FileNotFoundError(
                label_path
            )

        buckets[category].append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "label_path": label_path,
                "expected_boxes": (
                    record["person_boxes"]
                ),
            }
        )

    rng = random.Random(
        RANDOM_SEED
    )

    selected = []

    print()
    print("=" * 55)
    print("CrowdHuman stratified visual review")
    print("=" * 55)
    print()

    for category, target in (
        REVIEW_TARGETS.items()
    ):
        candidates = sorted(
            buckets[category],
            key=lambda x: x["image_id"],
        )

        if len(candidates) < target:
            raise RuntimeError(
                f"Not enough images in "
                f"{category}"
            )

        chosen = rng.sample(
            candidates,
            target,
        )

        for record in chosen:
            selected.append(
                (category, record)
            )

        print(
            f"{category}: "
            f"{target} selected from "
            f"{len(candidates)}"
        )

    print()
    print("Creating review images...")
    print()

    total_boxes = 0
    number = 1

    for category in REVIEW_TARGETS:
        category_records = [
            record
            for selected_category, record
            in selected
            if selected_category == category
        ]

        for record in category_records:
            output_path = (
                OUTPUT_DIR
                / (
                    f"{number:02d}_"
                    f"{category.replace('>', 'over')}_"
                    f"{record['image_id']}.jpg"
                )
            )

            box_count = draw_image(
                record["image_path"],
                record["label_path"],
                output_path,
            )

            if (
                box_count
                != record["expected_boxes"]
            ):
                raise RuntimeError(
                    f"Box mismatch for "
                    f"{record['image_id']}: "
                    f"expected "
                    f"{record['expected_boxes']}, "
                    f"got {box_count}"
                )

            total_boxes += box_count

            print(
                f"{number:02d}. "
                f"{category:<5} "
                f"{record['image_id']} "
                f"-> boxes={box_count}"
            )

            number += 1

    print()
    print("=" * 55)
    print("VISUAL CHECK CREATED")
    print("=" * 55)
    print()

    print(
        "Images reviewed:",
        len(selected),
    )

    print(
        "Boxes in reviewed images:",
        total_boxes,
    )

    print()
    print("Output:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()