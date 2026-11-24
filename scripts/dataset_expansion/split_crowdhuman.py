from pathlib import Path
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROWDHUMAN_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

IMAGES_DIR = CROWDHUMAN_ROOT / "images"
LABELS_DIR = CROWDHUMAN_ROOT / "labels"

METADATA_DIR = CROWDHUMAN_ROOT / "metadata"

MANIFEST_FILE = (
    METADATA_DIR
    / "selected_images.json"
)

OUTPUT_FILE = (
    METADATA_DIR
    / "crowdhuman_split.json"
)


RANDOM_SEED = 42


# 90% train / 10% validation.
#
# Split every person-count category separately
# so train and validation preserve the same
# CrowdHuman composition.
SPLIT_TARGETS = {
    "2-5": {
        "train": 360,
        "val": 40,
    },
    "6-10": {
        "train": 630,
        "val": 70,
    },
    "11-20": {
        "train": 630,
        "val": 70,
    },
    ">20": {
        "train": 180,
        "val": 20,
    },
}


def get_category(person_boxes):
    if 2 <= person_boxes <= 5:
        return "2-5"

    if 6 <= person_boxes <= 10:
        return "6-10"

    if 11 <= person_boxes <= 20:
        return "11-20"

    if person_boxes > 20:
        return ">20"

    raise ValueError(
        f"Unexpected person count: "
        f"{person_boxes}"
    )


def load_manifest():
    if not MANIFEST_FILE.exists():
        raise RuntimeError(
            f"Manifest not found: "
            f"{MANIFEST_FILE}"
        )

    data = json.loads(
        MANIFEST_FILE.read_text(
            encoding="utf-8"
        )
    )

    # selected_images.json may either contain
    # the list directly or store it under a
    # top-level key.
    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        if "records" in data:
            records = data["records"]
        elif "selected" in data:
            records = data["selected"]
        else:
            raise RuntimeError(
                "Cannot find selected image "
                "records in manifest"
            )

    else:
        raise RuntimeError(
            "Unexpected manifest format"
        )

    buckets = {
        category: []
        for category in SPLIT_TARGETS
    }

    for source_record in records:
        image_id = source_record.get(
            "image_id",
            source_record.get("ID"),
        )

        if image_id is None:
            raise RuntimeError(
                "Record without image_id"
            )

        image_id = str(image_id)

        # Prefer stored person count if present.
        person_boxes = source_record.get(
            "person_boxes"
        )

        if person_boxes is None:
            person_boxes = source_record.get(
                "valid_people"
            )

        # If the manifest does not store the
        # count explicitly, count YOLO boxes.
        label_path = (
            LABELS_DIR
            / f"{image_id}.txt"
        )

        image_path = (
            IMAGES_DIR
            / f"{image_id}.jpg"
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

        label_lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        actual_boxes = len(label_lines)

        if person_boxes is None:
            person_boxes = actual_boxes
        else:
            person_boxes = int(
                person_boxes
            )

            if person_boxes != actual_boxes:
                raise RuntimeError(
                    f"Box count mismatch for "
                    f"{image_id}: "
                    f"manifest={person_boxes}, "
                    f"label={actual_boxes}"
                )

        category = get_category(
            person_boxes
        )

        record = {
            "image_id": image_id,
            "category": category,
            "person_boxes": person_boxes,
        }

        buckets[category].append(
            record
        )

    return buckets


def split_bucket(
    records,
    train_count,
    val_count,
    seed,
):
    expected = (
        train_count
        + val_count
    )

    if len(records) != expected:
        raise RuntimeError(
            f"Expected {expected} records, "
            f"got {len(records)}"
        )

    records = sorted(
        records,
        key=lambda item: item["image_id"],
    )

    rng = random.Random(seed)
    rng.shuffle(records)

    train_records = records[
        :train_count
    ]

    val_records = records[
        train_count:
        train_count + val_count
    ]

    return train_records, val_records


def count_boxes(records):
    return sum(
        record["person_boxes"]
        for record in records
    )


def main():
    print()
    print(
        "======================================"
    )
    print(
        " CrowdHuman train/val split "
    )
    print(
        "======================================"
    )
    print()

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    buckets = load_manifest()

    train_records = []
    val_records = []

    category_summary = {}

    for index, (
        category,
        targets,
    ) in enumerate(
        SPLIT_TARGETS.items()
    ):
        records = buckets[category]

        category_seed = (
            RANDOM_SEED
            + index
        )

        category_train, category_val = (
            split_bucket(
                records=records,
                train_count=targets["train"],
                val_count=targets["val"],
                seed=category_seed,
            )
        )

        train_records.extend(
            category_train
        )

        val_records.extend(
            category_val
        )

        category_summary[category] = {
            "total": len(records),
            "train": len(
                category_train
            ),
            "val": len(
                category_val
            ),
            "train_boxes": count_boxes(
                category_train
            ),
            "val_boxes": count_boxes(
                category_val
            ),
        }

    train_records = sorted(
        train_records,
        key=lambda item: item["image_id"],
    )

    val_records = sorted(
        val_records,
        key=lambda item: item["image_id"],
    )

    train_ids = {
        record["image_id"]
        for record in train_records
    }

    val_ids = {
        record["image_id"]
        for record in val_records
    }

    overlap = (
        train_ids
        & val_ids
    )

    if overlap:
        raise RuntimeError(
            f"Train/val overlap: "
            f"{len(overlap)}"
        )

    if len(train_ids) != 1800:
        raise RuntimeError(
            f"Expected 1800 unique train "
            f"images, got {len(train_ids)}"
        )

    if len(val_ids) != 200:
        raise RuntimeError(
            f"Expected 200 unique val "
            f"images, got {len(val_ids)}"
        )

    all_ids = (
        train_ids
        | val_ids
    )

    if len(all_ids) != 2000:
        raise RuntimeError(
            f"Expected 2000 unique images, "
            f"got {len(all_ids)}"
        )

    train_boxes = count_boxes(
        train_records
    )

    val_boxes = count_boxes(
        val_records
    )

    total_boxes = (
        train_boxes
        + val_boxes
    )

    if total_boxes != 26873:
        raise RuntimeError(
            f"Expected 26873 boxes, "
            f"got {total_boxes}"
        )

    output = {
        "source": "CrowdHuman",
        "class_name": "person",
        "class_id": 0,
        "random_seed": RANDOM_SEED,
        "split_policy": (
            "stratified 90/10 by "
            "person-count category"
        ),
        "counts": {
            "total": 2000,
            "train": len(
                train_records
            ),
            "val": len(
                val_records
            ),
        },
        "boxes": {
            "total": total_boxes,
            "train": train_boxes,
            "val": val_boxes,
        },
        "categories": category_summary,
        "train": train_records,
        "val": val_records,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "======================================"
    )
    print("CATEGORY SPLIT")
    print(
        "======================================"
    )

    for (
        category,
        values,
    ) in category_summary.items():

        print()
        print(category)
        print(
            "  total:",
            values["total"],
        )
        print(
            "  train:",
            values["train"],
        )
        print(
            "  val:",
            values["val"],
        )
        print(
            "  train boxes:",
            values["train_boxes"],
        )
        print(
            "  val boxes:",
            values["val_boxes"],
        )

    print()
    print(
        "======================================"
    )
    print("FINAL SPLIT")
    print(
        "======================================"
    )
    print()

    print(
        "Train images:",
        len(train_records),
    )

    print(
        "Val images:",
        len(val_records),
    )

    print(
        "Total images:",
        len(all_ids),
    )

    print(
        "Train boxes:",
        train_boxes,
    )

    print(
        "Val boxes:",
        val_boxes,
    )

    print(
        "Total boxes:",
        total_boxes,
    )

    print(
        "Train/val overlap:",
        len(overlap),
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()