from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDD_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "pdd"
)

OPENIMAGES_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "openimages"
)

CROWDHUMAN_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "combined_split.json"
)


EXPECTED_COUNTS = {
    "train": 4215,
    "val": 532,
    "test": 132,
}

EXPECTED_SOURCE_COUNTS = {
    "train": {
        "pdd": 615,
        "openimages": 1800,
        "crowdhuman": 1800,
    },
    "val": {
        "pdd": 132,
        "openimages": 200,
        "crowdhuman": 200,
    },
    "test": {
        "pdd": 132,
        "openimages": 0,
        "crowdhuman": 0,
    },
}


def relative_path(path):
    return path.relative_to(
        PROJECT_ROOT
    ).as_posix()


def count_label_boxes(label_path):
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    return len(lines)


def validate_yolo_label(label_path):
    lines = [
        line.strip()
        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Bad YOLO label format: "
                f"{label_path}, "
                f"line {line_number}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise RuntimeError(
                f"Unexpected class "
                f"{class_id} in "
                f"{label_path}"
            )

        values = [
            float(value)
            for value in parts[1:]
        ]

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            raise RuntimeError(
                f"Out-of-range YOLO box: "
                f"{label_path}, "
                f"line {line_number}"
            )


def load_pdd():
    split_file = (
        PDD_ROOT
        / "split.json"
    )

    data = json.loads(
        split_file.read_text(
            encoding="utf-8"
        )
    )

    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for source_record in data["records"]:
        split = source_record["split"]

        if split not in records:
            raise RuntimeError(
                f"Unexpected PDD split: "
                f"{split}"
            )

        filename = source_record["file"]

        image_path = (
            PDD_ROOT
            / "images"
            / filename
        )

        label_path = (
            PDD_ROOT
            / "labels"
            / (
                Path(filename).stem
                + ".txt"
            )
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Missing PDD image: "
                f"{image_path}"
            )

        if not label_path.exists():
            raise RuntimeError(
                f"Missing PDD label: "
                f"{label_path}"
            )

        validate_yolo_label(
            label_path
        )

        actual_boxes = (
            count_label_boxes(
                label_path
            )
        )

        expected_boxes = int(
            source_record["objects"]
        )

        if actual_boxes != expected_boxes:
            raise RuntimeError(
                f"PDD box mismatch for "
                f"{filename}: "
                f"split.json={expected_boxes}, "
                f"label={actual_boxes}"
            )

        record = {
            "uid": (
                "pdd:"
                + filename
            ),
            "source": "pdd",
            "image_id": filename,
            "image_path": relative_path(
                image_path
            ),
            "label_path": relative_path(
                label_path
            ),
            "person_boxes": actual_boxes,
            "width": int(
                source_record["width"]
            ),
            "height": int(
                source_record["height"]
            ),
            "group": source_record[
                "group"
            ],
            "sha256": source_record[
                "sha256"
            ],
        }

        records[split].append(
            record
        )

    return records


def load_external_source(
    source_name,
    root,
    split_filename,
):
    split_file = (
        root
        / "metadata"
        / split_filename
    )

    data = json.loads(
        split_file.read_text(
            encoding="utf-8"
        )
    )

    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for split in (
        "train",
        "val",
    ):
        for source_record in data[split]:
            image_id = str(
                source_record["image_id"]
            )

            image_path = (
                root
                / "images"
                / f"{image_id}.jpg"
            )

            label_path = (
                root
                / "labels"
                / f"{image_id}.txt"
            )

            if not image_path.exists():
                raise RuntimeError(
                    f"Missing {source_name} "
                    f"image: {image_path}"
                )

            if not label_path.exists():
                raise RuntimeError(
                    f"Missing {source_name} "
                    f"label: {label_path}"
                )

            validate_yolo_label(
                label_path
            )

            actual_boxes = (
                count_label_boxes(
                    label_path
                )
            )

            expected_boxes = int(
                source_record[
                    "person_boxes"
                ]
            )

            if actual_boxes != expected_boxes:
                raise RuntimeError(
                    f"{source_name} box "
                    f"mismatch for "
                    f"{image_id}: "
                    f"manifest="
                    f"{expected_boxes}, "
                    f"label="
                    f"{actual_boxes}"
                )

            record = {
                "uid": (
                    source_name
                    + ":"
                    + image_id
                ),
                "source": source_name,
                "image_id": image_id,
                "image_path": relative_path(
                    image_path
                ),
                "label_path": relative_path(
                    label_path
                ),
                "person_boxes": (
                    actual_boxes
                ),
            }

            if "category" in source_record:
                record["category"] = (
                    source_record[
                        "category"
                    ]
                )

            records[split].append(
                record
            )

    return records


def count_sources(records):
    counts = {
        "pdd": 0,
        "openimages": 0,
        "crowdhuman": 0,
    }

    for record in records:
        counts[
            record["source"]
        ] += 1

    return counts


def count_boxes(records):
    return sum(
        record["person_boxes"]
        for record in records
    )


def validate_unique_uids(
    combined,
):
    split_uids = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        uids = [
            record["uid"]
            for record
            in combined[split]
        ]

        unique = set(uids)

        if len(unique) != len(uids):
            raise RuntimeError(
                f"Duplicate UID inside "
                f"{split}: "
                f"{len(uids) - len(unique)}"
            )

        split_uids[split] = unique

    train_val = (
        split_uids["train"]
        & split_uids["val"]
    )

    train_test = (
        split_uids["train"]
        & split_uids["test"]
    )

    val_test = (
        split_uids["val"]
        & split_uids["test"]
    )

    if train_val:
        raise RuntimeError(
            f"Train/val overlap: "
            f"{len(train_val)}"
        )

    if train_test:
        raise RuntimeError(
            f"Train/test overlap: "
            f"{len(train_test)}"
        )

    if val_test:
        raise RuntimeError(
            f"Val/test overlap: "
            f"{len(val_test)}"
        )

    return {
        "train_val": len(
            train_val
        ),
        "train_test": len(
            train_test
        ),
        "val_test": len(
            val_test
        ),
    }


def main():
    print()
    print(
        "======================================"
    )
    print(
        " Combined dataset manifest "
    )
    print(
        "======================================"
    )
    print()

    pdd = load_pdd()

    openimages = (
        load_external_source(
            source_name="openimages",
            root=OPENIMAGES_ROOT,
            split_filename=(
                "openimages_split.json"
            ),
        )
    )

    crowdhuman = (
        load_external_source(
            source_name="crowdhuman",
            root=CROWDHUMAN_ROOT,
            split_filename=(
                "crowdhuman_split.json"
            ),
        )
    )

    combined = {
        "train": (
            pdd["train"]
            + openimages["train"]
            + crowdhuman["train"]
        ),
        "val": (
            pdd["val"]
            + openimages["val"]
            + crowdhuman["val"]
        ),
        "test": (
            pdd["test"]
        ),
    }

    for split in combined:
        combined[split] = sorted(
            combined[split],
            key=lambda item: (
                item["source"],
                item["image_id"],
            ),
        )

    for split, expected in (
        EXPECTED_COUNTS.items()
    ):
        actual = len(
            combined[split]
        )

        if actual != expected:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected} images, "
                f"got {actual}"
            )

        source_counts = (
            count_sources(
                combined[split]
            )
        )

        if (
            source_counts
            != EXPECTED_SOURCE_COUNTS[
                split
            ]
        ):
            raise RuntimeError(
                f"{split}: unexpected "
                f"source counts: "
                f"{source_counts}"
            )

    overlaps = (
        validate_unique_uids(
            combined
        )
    )

    # Critical benchmark protection:
    # test must contain only PDD.
    test_sources = {
        record["source"]
        for record
        in combined["test"]
    }

    if test_sources != {"pdd"}:
        raise RuntimeError(
            "Final test split must "
            "contain only PDD"
        )

    pdd_test_uids = {
        record["uid"]
        for record
        in pdd["test"]
    }

    combined_test_uids = {
        record["uid"]
        for record
        in combined["test"]
    }

    if (
        pdd_test_uids
        != combined_test_uids
    ):
        raise RuntimeError(
            "PDD test set was modified"
        )

    source_summary = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        source_summary[split] = (
            count_sources(
                combined[split]
            )
        )

    box_summary = {
        split: count_boxes(
            combined[split]
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    output = {
        "name": (
            "RescueVision combined "
            "person detection dataset"
        ),
        "class_name": "person",
        "class_id": 0,
        "policy": {
            "train": (
                "PDD train + "
                "Open Images train + "
                "CrowdHuman train"
            ),
            "val": (
                "PDD val + "
                "Open Images val + "
                "CrowdHuman val"
            ),
            "test": (
                "PDD test only; "
                "target-domain benchmark"
            ),
        },
        "counts": {
            split: len(
                combined[split]
            )
            for split in (
                "train",
                "val",
                "test",
            )
        },
        "boxes": box_summary,
        "sources": source_summary,
        "overlap": overlaps,
        "train": combined["train"],
        "val": combined["val"],
        "test": combined["test"],
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("TRAIN")
    print(
        "  PDD:",
        source_summary[
            "train"
        ]["pdd"],
    )
    print(
        "  Open Images:",
        source_summary[
            "train"
        ]["openimages"],
    )
    print(
        "  CrowdHuman:",
        source_summary[
            "train"
        ]["crowdhuman"],
    )
    print(
        "  Total:",
        len(combined["train"]),
    )
    print(
        "  Boxes:",
        box_summary["train"],
    )

    print()
    print("VAL")
    print(
        "  PDD:",
        source_summary[
            "val"
        ]["pdd"],
    )
    print(
        "  Open Images:",
        source_summary[
            "val"
        ]["openimages"],
    )
    print(
        "  CrowdHuman:",
        source_summary[
            "val"
        ]["crowdhuman"],
    )
    print(
        "  Total:",
        len(combined["val"]),
    )
    print(
        "  Boxes:",
        box_summary["val"],
    )

    print()
    print("TEST")
    print(
        "  PDD:",
        source_summary[
            "test"
        ]["pdd"],
    )
    print(
        "  Open Images:",
        source_summary[
            "test"
        ]["openimages"],
    )
    print(
        "  CrowdHuman:",
        source_summary[
            "test"
        ]["crowdhuman"],
    )
    print(
        "  Total:",
        len(combined["test"]),
    )
    print(
        "  Boxes:",
        box_summary["test"],
    )

    print()
    print("OVERLAP")
    print(
        "  train/val:",
        overlaps["train_val"],
    )
    print(
        "  train/test:",
        overlaps["train_test"],
    )
    print(
        "  val/test:",
        overlaps["val_test"],
    )

    print()
    print(
        "PDD test preserved:",
        "YES",
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()