import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "combined_split.json"
)

OUTPUT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "combined_yolo"
)


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
        return "link"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main():
    data = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    total_links = 0
    total_copies = 0

    for split in ("train", "val", "test"):
        images_dir = OUTPUT / "images" / split
        labels_dir = OUTPUT / "labels" / split

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        print(f"\n{split.upper()}")

        for index, record in enumerate(
            data[split],
            start=1,
        ):
            source_image = (
                PROJECT_ROOT / record["image_path"]
            )
            source_label = (
                PROJECT_ROOT / record["label_path"]
            )

            # Unique filename prevents collisions between
            # PDD/OpenImages/CrowdHuman/Humans Disaster.
            source_name = record["source"]
            uid = f"{source_name}_{index:06d}"

            image_destination = (
                images_dir
                / f"{uid}{source_image.suffix.lower()}"
            )

            label_destination = (
                labels_dir / f"{uid}.txt"
            )

            result = link_or_copy(
                source_image,
                image_destination,
            )

            if result == "link":
                total_links += 1
            else:
                total_copies += 1

            result = link_or_copy(
                source_label,
                label_destination,
            )

            if result == "link":
                total_links += 1
            else:
                total_copies += 1

        print(
            f"  Images: {len(data[split])}"
        )

    yaml_path = OUTPUT / "data.yaml"

    yaml_path.write_text(
        "\n".join([
            f"path: {OUTPUT.resolve().as_posix()}",
            "",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "nc: 1",
            "",
            "names:",
            "  0: person",
            "",
        ]),
        encoding="utf-8",
    )

    print("\n================================")
    print("Combined YOLO dataset ready")
    print("================================")
    print("Hard links:", total_links)
    print("Copies:", total_copies)
    print()
    print("YAML:")
    print(yaml_path.resolve())


if __name__ == "__main__":
    main()