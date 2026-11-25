from pathlib import Path
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

METADATA = ROOT / "metadata"

ANNOTATION_FILE = (
    METADATA
    / "annotation_train.odgt"
)

OUTPUT_FILE = (
    METADATA
    / "selected_images.json"
)

DOWNLOAD_LIST = (
    METADATA
    / "download_ids.txt"
)

RANDOM_SEED = 42

TARGETS = {
    "2-5": 400,
    "6-10": 700,
    "11-20": 700,
    ">20": 200,
}


def is_ignored(box):
    extra = box.get("extra", {})
    return int(extra.get("ignore", 0)) == 1


def valid_fbox(box):
    fbox = box.get("fbox")

    if not isinstance(fbox, list):
        return False

    if len(fbox) != 4:
        return False

    x, y, w, h = fbox

    return w > 0 and h > 0


def category_for(count):
    if 2 <= count <= 5:
        return "2-5"

    if 6 <= count <= 10:
        return "6-10"

    if 11 <= count <= 20:
        return "11-20"

    if count > 20:
        return ">20"

    return None


def load_records():
    buckets = {
        category: []
        for category in TARGETS
    }

    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            image_id = record["ID"]

            valid_boxes = []

            for box in record.get(
                "gtboxes",
                [],
            ):
                if is_ignored(box):
                    continue

                if not valid_fbox(box):
                    continue

                valid_boxes.append(
                    box["fbox"]
                )

            count = len(valid_boxes)

            category = category_for(
                count
            )

            if category is None:
                continue

            buckets[category].append(
                {
                    "image_id": image_id,
                    "category": category,
                    "person_boxes": count,
                    "fboxes": valid_boxes,
                }
            )

    return buckets


def main():
    print()
    print("=" * 50)
    print("CrowdHuman sample selection")
    print("=" * 50)

    print()
    print("Random seed:", RANDOM_SEED)

    buckets = load_records()

    print()
    print("=" * 50)
    print("AVAILABLE")
    print("=" * 50)

    for category, records in buckets.items():
        print(
            f"{category}: {len(records)}"
        )

    rng = random.Random(
        RANDOM_SEED
    )

    selected = []

    print()
    print("=" * 50)
    print("SELECTED")
    print("=" * 50)

    for category, target in TARGETS.items():
        records = sorted(
            buckets[category],
            key=lambda x: x["image_id"],
        )

        if len(records) < target:
            raise RuntimeError(
                f"Not enough images in "
                f"{category}: "
                f"{len(records)} < {target}"
            )

        chosen = rng.sample(
            records,
            target,
        )

        selected.extend(chosen)

        print(
            f"{category}: {len(chosen)}"
        )

    selected = sorted(
        selected,
        key=lambda x: x["image_id"],
    )

    ids = [
        record["image_id"]
        for record in selected
    ]

    unique_ids = set(ids)

    if len(selected) != 2000:
        raise RuntimeError(
            "Expected 2000 selected images"
        )

    if len(unique_ids) != 2000:
        raise RuntimeError(
            "Duplicate image IDs detected"
        )

    total_boxes = sum(
        record["person_boxes"]
        for record in selected
    )

    output = {
        "source": "CrowdHuman",
        "source_split": "train",
        "class_name": "person",
        "class_id": 0,
        "box_type": "fbox",
        "ignored_boxes_removed": True,
        "random_seed": RANDOM_SEED,
        "targets": TARGETS,
        "images": len(selected),
        "person_boxes": total_boxes,
        "records": selected,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    DOWNLOAD_LIST.write_text(
        "\n".join(ids) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 50)
    print("FINAL")
    print("=" * 50)

    print()
    print(
        "Selected images:",
        len(selected),
    )

    print(
        "Unique IDs:",
        len(unique_ids),
    )

    print(
        "Selected person boxes:",
        total_boxes,
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)
    print(DOWNLOAD_LIST)


if __name__ == "__main__":
    main()