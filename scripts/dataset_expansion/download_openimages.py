from pathlib import Path
import csv
import random
import sys
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "openimages"
)

IMAGES_DIR = OUTPUT_ROOT / "images"
LABELS_DIR = OUTPUT_ROOT / "labels"
METADATA_DIR = OUTPUT_ROOT / "metadata"

for directory in [
    OUTPUT_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    METADATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


PERSON_LABEL = "/m/01g317"

# Final Open Images subset used for dataset expansion.
TARGET_IMAGES = 2000

RANDOM_SEED = 42
FINAL_TARGETS = {
    "single": 800,
    "small_group": 900,
    "medium_group": 300,
    "crowd": 0,
}

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/openimages/v6/"
    "oidv6-train-annotations-bbox.csv"
)

ANNOTATIONS_FILE = (
    METADATA_DIR
    / "oidv6-train-annotations-bbox.csv"
)


def download_file(url: str, destination: Path):
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return

    print()
    print("Downloading:")
    print(url)
    print()
    print("Saving to:")
    print(destination)
    print()

    with requests.get(
        url,
        stream=True,
        timeout=60,
    ) as response:

        response.raise_for_status()

        total = int(
            response.headers.get("content-length", 0)
        )

        downloaded = 0

        with destination.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                file.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = downloaded / total * 100

                    print(
                        f"\rDownloaded: {percent:6.2f}%",
                        end="",
                        flush=True,
                    )

    print()
    print("Download complete.")


def as_flag(row, name):
    """
    Open Images flags are stored as 0/1.
    If a column is absent, return 0.
    """
    value = row.get(name, "0")

    if value in (None, ""):
        return 0

    return int(float(value))


def valid_box(xmin, xmax, ymin, ymax):
    return (
        0.0 <= xmin < xmax <= 1.0
        and 0.0 <= ymin < ymax <= 1.0
    )


def yolo_box(xmin, xmax, ymin, ymax):
    width = xmax - xmin
    height = ymax - ymin

    x_center = xmin + width / 2.0
    y_center = ymin + height / 2.0

    return x_center, y_center, width, height


def collect_person_annotations(csv_path: Path):
    """
    Collect usable individual Person annotations.

    Keep:
      - normal people
      - occluded people
      - truncated people

    Remove:
      - GroupOf boxes
      - depictions of people
      - invalid boxes
    """

    persons = {}

    stats = {
        "person_rows": 0,
        "kept_boxes": 0,
        "group_of_removed": 0,
        "depiction_removed": 0,
        "invalid_removed": 0,
        "occluded_kept": 0,
        "truncated_kept": 0,
    }

    print()
    print("Scanning annotations for usable Person boxes...")
    print()

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        print(
            "CSV columns:",
            ", ".join(reader.fieldnames or []),
        )
        print()

        for row in reader:

            if row["LabelName"] != PERSON_LABEL:
                continue

            stats["person_rows"] += 1

            is_group_of = as_flag(
                row,
                "IsGroupOf",
            )

            is_depiction = as_flag(
                row,
                "IsDepiction",
            )

            is_occluded = as_flag(
                row,
                "IsOccluded",
            )

            is_truncated = as_flag(
                row,
                "IsTruncated",
            )

            if is_group_of == 1:
                stats["group_of_removed"] += 1
                continue

            if is_depiction == 1:
                stats["depiction_removed"] += 1
                continue

            image_id = row["ImageID"]

            xmin = float(row["XMin"])
            xmax = float(row["XMax"])
            ymin = float(row["YMin"])
            ymax = float(row["YMax"])

            if not valid_box(
                xmin,
                xmax,
                ymin,
                ymax,
            ):
                stats["invalid_removed"] += 1
                continue

            if is_occluded == 1:
                stats["occluded_kept"] += 1

            if is_truncated == 1:
                stats["truncated_kept"] += 1

            annotation = {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "is_occluded": is_occluded,
                "is_truncated": is_truncated,
            }

            persons.setdefault(
                image_id,
                [],
            ).append(annotation)

            stats["kept_boxes"] += 1

    return persons, stats


def create_buckets(persons):
    """
    Divide images according to the number
    of usable individual Person boxes.
    """

    buckets = {
        "single": [],
        "small_group": [],
        "medium_group": [],
        "crowd": [],
    }

    for image_id, boxes in persons.items():
        count = len(boxes)

        if count == 1:
            buckets["single"].append(image_id)

        elif 2 <= count <= 4:
            buckets["small_group"].append(image_id)

        elif 5 <= count <= 10:
            buckets["medium_group"].append(image_id)

        elif count >= 11:
            buckets["crowd"].append(image_id)

    return buckets


def select_final_images(buckets):
    """
    Create the final deterministic Open Images subset.

    Distribution:
      - 650 single-person images
      - 800 small groups (2-4 people)
      - 450 medium groups (5-10 people)
      - 100 crowds (11+ individually annotated people)

    The fixed random seed makes the selection reproducible.
    """

    rng = random.Random(RANDOM_SEED)

    selected = []
    selected_by_bucket = {}

    for bucket_name, target in FINAL_TARGETS.items():
        candidates = sorted(buckets[bucket_name])

        if len(candidates) < target:
            print()
            print(
                f"ERROR: bucket '{bucket_name}' has only "
                f"{len(candidates)} images, but {target} are required."
            )
            sys.exit(1)

        rng.shuffle(candidates)

        chosen = candidates[:target]

        selected.extend(chosen)
        selected_by_bucket[bucket_name] = chosen

    rng.shuffle(selected)

    return selected, selected_by_bucket


def clear_old_labels():
    for path in LABELS_DIR.glob("*.txt"):
        path.unlink()


def write_labels(persons, image_ids):
    clear_old_labels()

    for image_id in image_ids:

        output_file = (
            LABELS_DIR
            / f"{image_id}.txt"
        )

        lines = []

        for annotation in persons[image_id]:

            xc, yc, w, h = yolo_box(
                annotation["xmin"],
                annotation["xmax"],
                annotation["ymin"],
                annotation["ymax"],
            )

            lines.append(
                "0 "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

        output_file.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def write_manifest(persons, image_ids):
    manifest = (
        METADATA_DIR
        / "selected_images.csv"
    )

    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "image_id",
                "category",
                "person_boxes",
                "occluded_boxes",
                "truncated_boxes",
            ]
        )

        for image_id in image_ids:

            boxes = persons[image_id]

            occluded = sum(
                box["is_occluded"]
                for box in boxes
            )

            truncated = sum(
                box["is_truncated"]
                for box in boxes
            )

            person_count = len(boxes)
            
            if person_count == 1:
                category = "single"
            elif 2 <= person_count <= 4:
                category = "small_group"
            elif 5 <= person_count <= 10:
                category = "medium_group"
            else:
                category = "crowd"
            
            writer.writerow(
      		    [
        	        image_id,
        	        category,
                    person_count,
                    occluded,
                    truncated,
    		    ]
            )

    return manifest


def write_download_list(image_ids):
    download_list = (
        METADATA_DIR
        / "download_list.txt"
    )

    with download_list.open(
        "w",
        encoding="utf-8",
    ) as file:

        for image_id in image_ids:
            file.write(
                f"train/{image_id}\n"
            )

    return download_list


def main():

    print()
    print(
        "======================================"
    )
    print(
        " RescueVisionLab - Open Images filter "
    )
    print(
        "======================================"
    )

    print()
    print("Person label:", PERSON_LABEL)
    print("Final target:", TARGET_IMAGES)
    print("Random seed:", RANDOM_SEED)

    download_file(
        ANNOTATIONS_URL,
        ANNOTATIONS_FILE,
    )

    persons, stats = collect_person_annotations(
        ANNOTATIONS_FILE
    )

    print(
        "======================================"
    )
    print("ANNOTATION FILTER")
    print(
        "======================================"
    )

    print(
        "All Person rows:",
        stats["person_rows"],
    )

    print(
        "Kept individual boxes:",
        stats["kept_boxes"],
    )

    print(
        "Removed GroupOf:",
        stats["group_of_removed"],
    )

    print(
        "Removed Depiction:",
        stats["depiction_removed"],
    )

    print(
        "Removed invalid:",
        stats["invalid_removed"],
    )

    print(
        "Kept occluded:",
        stats["occluded_kept"],
    )

    print(
        "Kept truncated:",
        stats["truncated_kept"],
    )

    print()
    print(
        "Images after filtering:",
        len(persons),
    )

    buckets = create_buckets(persons)

    print()
    print(
        "======================================"
    )
    print("AVAILABLE IMAGES")
    print(
        "======================================"
    )

    for name, values in buckets.items():
        print(
            f"{name}: {len(values)}"
        )

    selected, selected_by_bucket = (
        select_final_images(buckets)
    )

    if len(selected) != TARGET_IMAGES:
        print()
        print(
            "ERROR: expected",
            TARGET_IMAGES,
            "selected images, got",
            len(selected),
        )
        sys.exit(1)

    print()
    print(
        "======================================"
    )
    print("FINAL SAMPLE")
    print(
        "======================================"
    )

    for name, values in selected_by_bucket.items():
        print(
            f"{name}: {len(values)}"
        )

    write_labels(
        persons,
        selected,
    )

    manifest = write_manifest(
        persons,
        selected,
    )

    download_list = write_download_list(
        selected
    )

    total_boxes = sum(
        len(persons[image_id])
        for image_id in selected
    )

    print()
    print(
        "Selected images:",
        len(selected),
    )

    print(
        "Selected individual boxes:",
        total_boxes,
    )

    print()
    print("Manifest:")
    print(manifest)

    print()
    print("Download list:")
    print(download_list)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Old JPG files are NOT automatically deleted."
    )
    print(
        "The new download list may contain different IDs."
    )


if __name__ == "__main__":
    main()