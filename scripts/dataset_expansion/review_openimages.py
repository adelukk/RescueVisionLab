from pathlib import Path
import csv
import random

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "openimages"
)

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
METADATA_DIR = DATASET_ROOT / "metadata"

MANIFEST_FILE = (
    METADATA_DIR
    / "selected_images.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "openimages_visual_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42

REVIEW_TARGETS = {
    "single": 8,
    "small_group": 8,
    "medium_group": 8,
}


def get_bucket(person_boxes):
    if person_boxes == 1:
        return "single"

    if 2 <= person_boxes <= 4:
        return "small_group"

    if 5 <= person_boxes <= 10:
        return "medium_group"

    if person_boxes >= 11:
        return "crowd"

    raise ValueError(
        f"Invalid person box count: {person_boxes}"
    )


def read_yolo_labels(label_path):
    boxes = []

    with label_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}: "
                    f"invalid line {line_number}: "
                    f"{line}"
                )

            class_id = int(parts[0])

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            if class_id != 0:
                raise ValueError(
                    f"{label_path}: "
                    f"unexpected class {class_id}"
                )

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < bw <= 1
                and 0 < bh <= 1
            ):
                raise ValueError(
                    f"{label_path}: "
                    f"invalid normalized box"
                )

            boxes.append(
                (
                    class_id,
                    xc,
                    yc,
                    bw,
                    bh,
                )
            )

    return boxes


def yolo_to_xyxy(
    xc,
    yc,
    bw,
    bh,
    image_width,
    image_height,
):
    x1 = (xc - bw / 2) * image_width
    y1 = (yc - bh / 2) * image_height

    x2 = (xc + bw / 2) * image_width
    y2 = (yc + bh / 2) * image_height

    x1 = max(
        0,
        min(image_width - 1, x1),
    )
    y1 = max(
        0,
        min(image_height - 1, y1),
    )
    x2 = max(
        0,
        min(image_width - 1, x2),
    )
    y2 = max(
        0,
        min(image_height - 1, y2),
    )

    return x1, y1, x2, y2


def draw_sample(
    image_path,
    label_path,
    output_path,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    boxes = read_yolo_labels(
        label_path
    )

    line_width = max(
        2,
        round(min(width, height) / 250),
    )

    for index, (
        _,
        xc,
        yc,
        bw,
        bh,
    ) in enumerate(
        boxes,
        start=1,
    ):
        x1, y1, x2, y2 = yolo_to_xyxy(
            xc,
            yc,
            bw,
            bh,
            width,
            height,
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=line_width,
        )

        text = f"PERSON {index}"

        text_y = max(
            0,
            y1 - 15,
        )

        draw.text(
            (x1 + 2, text_y),
            text,
            fill="lime",
        )

    image.save(
        output_path,
        quality=92,
    )

    return len(boxes)


def load_manifest():
    if not MANIFEST_FILE.exists():
        raise RuntimeError(
            f"Manifest not found: "
            f"{MANIFEST_FILE}"
        )

    buckets = {
        name: []
        for name in REVIEW_TARGETS
    }

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            image_id = row["image_id"]

            person_boxes = int(
                row["person_boxes"]
            )

            bucket = get_bucket(
                person_boxes
            )

            image_path = (
                IMAGES_DIR
                / f"{image_id}.jpg"
            )

            label_path = (
                LABELS_DIR
                / f"{image_id}.txt"
            )

            if not image_path.exists():
                raise RuntimeError(
                    f"Missing image: "
                    f"{image_path}"
                )

            if not label_path.exists():
                raise RuntimeError(
                    f"Missing label: "
                    f"{label_path}"
                )

            buckets[bucket].append(
                (
                    image_path,
                    label_path,
                    person_boxes,
                )
            )

    return buckets


def main():
    buckets = load_manifest()

    random_generator = random.Random(
        RANDOM_SEED
    )

    selected = []

    print()
    print(
        "======================================"
    )
    print(
        " Open Images stratified visual review "
    )
    print(
        "======================================"
    )
    print()

    for bucket_name, target in (
        REVIEW_TARGETS.items()
    ):
        candidates = sorted(
            buckets[bucket_name],
            key=lambda item: item[0].stem,
        )

        if len(candidates) < target:
            raise RuntimeError(
                f"Not enough images in "
                f"{bucket_name}: "
                f"{len(candidates)} < {target}"
            )

        chosen = random_generator.sample(
            candidates,
            target,
        )

        for item in chosen:
            selected.append(
                (
                    bucket_name,
                    *item,
                )
            )

        print(
            f"{bucket_name}: "
            f"{target} selected "
            f"from {len(candidates)}"
        )

    # Remove images from the previous visual review.
    for old_file in OUTPUT_DIR.glob(
        "*.jpg"
    ):
        old_file.unlink()

    total_boxes = 0

    print()
    print(
        "Creating review images..."
    )
    print()

    number = 1

    for bucket_name in REVIEW_TARGETS:

        bucket_items = [
            item
            for item in selected
            if item[0] == bucket_name
        ]

        for (
            _,
            image_path,
            label_path,
            expected_boxes,
        ) in bucket_items:

            output_path = (
                OUTPUT_DIR
                / (
                    f"{number:02d}_"
                    f"{bucket_name}_"
                    f"{image_path.stem}.jpg"
                )
            )

            box_count = draw_sample(
                image_path,
                label_path,
                output_path,
            )

            if box_count != expected_boxes:
                raise RuntimeError(
                    f"Box count mismatch for "
                    f"{image_path.name}: "
                    f"manifest={expected_boxes}, "
                    f"label={box_count}"
                )

            total_boxes += box_count

            print(
                f"{number:02d}. "
                f"{bucket_name:<12} "
                f"{image_path.name} "
                f"-> boxes={box_count}"
            )

            number += 1

    print()
    print(
        "======================================"
    )
    print(
        "VISUAL CHECK CREATED"
    )
    print(
        "======================================"
    )
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