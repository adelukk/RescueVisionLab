from pathlib import Path
import json

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
METADATA = ROOT / "metadata"

SELECTION_FILE = (
    METADATA
    / "selected_images.json"
)


def clamp(value, low, high):
    return max(low, min(value, high))


def convert_box(fbox, image_width, image_height):
    x, y, w, h = map(float, fbox)

    # CrowdHuman fbox is [x, y, width, height].
    x1 = clamp(x, 0.0, float(image_width))
    y1 = clamp(y, 0.0, float(image_height))

    x2 = clamp(
        x + w,
        0.0,
        float(image_width),
    )

    y2 = clamp(
        y + h,
        0.0,
        float(image_height),
    )

    clipped_w = x2 - x1
    clipped_h = y2 - y1

    if clipped_w <= 0 or clipped_h <= 0:
        return None

    x_center = (
        (x1 + x2) / 2.0
    ) / image_width

    y_center = (
        (y1 + y2) / 2.0
    ) / image_height

    norm_w = (
        clipped_w / image_width
    )

    norm_h = (
        clipped_h / image_height
    )

    return (
        x_center,
        y_center,
        norm_w,
        norm_h,
    )


def main():
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            SELECTION_FILE
        )

    LABELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    records = data["records"]

    print()
    print("=" * 60)
    print("CrowdHuman -> YOLO conversion")
    print("=" * 60)

    print()
    print("Selected records:", len(records))

    images_processed = 0
    boxes_input = 0
    boxes_written = 0
    boxes_clipped = 0
    boxes_removed = 0

    for record in records:
        image_id = record["image_id"]

        image_path = (
            IMAGES
            / f"{image_id}.jpg"
        )

        label_path = (
            LABELS
            / f"{image_id}.txt"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        if image_width <= 0 or image_height <= 0:
            raise RuntimeError(
                f"Invalid image size: "
                f"{image_path}"
            )

        output_lines = []

        for fbox in record["fboxes"]:
            boxes_input += 1

            x, y, w, h = map(
                float,
                fbox,
            )

            original = (
                x,
                y,
                x + w,
                y + h,
            )

            converted = convert_box(
                fbox,
                image_width,
                image_height,
            )

            if converted is None:
                boxes_removed += 1
                continue

            clipped = (
                clamp(
                    original[0],
                    0.0,
                    float(image_width),
                ),
                clamp(
                    original[1],
                    0.0,
                    float(image_height),
                ),
                clamp(
                    original[2],
                    0.0,
                    float(image_width),
                ),
                clamp(
                    original[3],
                    0.0,
                    float(image_height),
                ),
            )

            if any(
                abs(a - b) > 1e-9
                for a, b in zip(
                    original,
                    clipped,
                )
            ):
                boxes_clipped += 1

            xc, yc, nw, nh = converted

            output_lines.append(
                "0 "
                f"{xc:.8f} "
                f"{yc:.8f} "
                f"{nw:.8f} "
                f"{nh:.8f}"
            )

            boxes_written += 1

        if not output_lines:
            raise RuntimeError(
                f"No usable boxes for "
                f"{image_id}"
            )

        label_path.write_text(
            "\n".join(output_lines) + "\n",
            encoding="utf-8",
        )

        images_processed += 1

    jpg_ids = {
        p.stem
        for p in IMAGES.glob("*.jpg")
    }

    txt_ids = {
        p.stem
        for p in LABELS.glob("*.txt")
    }

    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)

    print()
    print(
        "Images processed:",
        images_processed,
    )

    print(
        "Input boxes:",
        boxes_input,
    )

    print(
        "Written boxes:",
        boxes_written,
    )

    print(
        "Clipped boxes:",
        boxes_clipped,
    )

    print(
        "Removed boxes:",
        boxes_removed,
    )

    print()
    print(
        "JPG:",
        len(jpg_ids),
    )

    print(
        "TXT:",
        len(txt_ids),
    )

    print(
        "Matched:",
        len(jpg_ids & txt_ids),
    )

    print(
        "JPG without TXT:",
        len(jpg_ids - txt_ids),
    )

    print(
        "TXT without JPG:",
        len(txt_ids - jpg_ids),
    )

    if images_processed != 2000:
        raise RuntimeError(
            f"Expected 2000 images, "
            f"got {images_processed}"
        )

    if jpg_ids != txt_ids:
        raise RuntimeError(
            "JPG/TXT correspondence failed"
        )

    print()
    print(
        "CrowdHuman YOLO labels are ready."
    )


if __name__ == "__main__":
    main()