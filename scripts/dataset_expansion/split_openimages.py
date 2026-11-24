from pathlib import Path
import csv
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OPENIMAGES_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "openimages"
)

IMAGES_DIR = OPENIMAGES_ROOT / "images"
LABELS_DIR = OPENIMAGES_ROOT / "labels"

METADATA_DIR = OPENIMAGES_ROOT / "metadata"

MANIFEST_FILE = (
    METADATA_DIR
    / "selected_images.csv"
)

OUTPUT_FILE = (
    METADATA_DIR
    / "openimages_split.json"
)


RANDOM_SEED = 42


# 90% train / 10% validation.
#
# We split each category separately so that
# train and validation have the same composition.
SPLIT_TARGETS = {
    "single": {
        "train": 720,
        "val": 80,
    },
    "small_group": {
        "train": 810,
        "val": 90,
    },
    "medium_group": {
        "train": 270,
        "val": 30,
    },
}


def load_manifest():
    if not MANIFEST_FILE.exists():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    buckets = {
        category: []
        for category in SPLIT_TARGETS
    }

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "image_id",
            "category",
            "person_boxes",
        }

        actual_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - actual_columns
        )

        if missing_columns:
            raise RuntimeError(
                "Manifest is missing columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row in reader:
            image_id = row["image_id"]
            category = row["category"]

            if category not in buckets:
                raise RuntimeError(
                    f"Unexpected category: "
                    f"{category}"
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

            record = {
                "image_id": image_id,
                "category": category,
                "person_boxes": int(
                    row["person_boxes"]
                ),
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
        " Open Images train/val split "
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

        # Different deterministic seed for
        # every category.
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

    # Stable final ordering in the JSON.
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

    output = {
        "source": "Open Images",
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
            "total": (
                count_boxes(
                    train_records
                )
                + count_boxes(
                    val_records
                )
            ),
            "train": count_boxes(
                train_records
            ),
            "val": count_boxes(
                val_records
            ),
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
        count_boxes(
            train_records
        ),
    )

    print(
        "Val boxes:",
        count_boxes(
            val_records
        ),
    )

    print(
        "Total boxes:",
        (
            count_boxes(
                train_records
            )
            + count_boxes(
                val_records
            )
        ),
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