from pathlib import Path
import json
import shutil
import zipfile

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

METADATA = ROOT / "metadata"
IMAGES = ROOT / "images"
TEMP = ROOT / "_temp"

SELECTION_FILE = (
    METADATA
    / "selected_images.json"
)

REPO_ID = "sshao0516/CrowdHuman"

ARCHIVES = [
    "CrowdHuman_train01.zip",
    "CrowdHuman_train02.zip",
    "CrowdHuman_train03.zip",
]


def load_selected_ids():
    data = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    ids = {
        record["image_id"]
        for record in data["records"]
    }

    if len(ids) != 2000:
        raise RuntimeError(
            f"Expected 2000 selected IDs, "
            f"got {len(ids)}"
        )

    return ids


def normalize_image_id(filename):
    return Path(filename).stem


def process_archive(
    archive_name,
    selected_ids,
):
    print()
    print("=" * 60)
    print("ARCHIVE:", archive_name)
    print("=" * 60)

    print()
    print("Downloading...")

    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=archive_name,
            local_dir=str(TEMP),
        )
    )

    print("Downloaded:")
    print(archive_path)

    extracted = 0
    already_exists = 0

    print()
    print("Scanning archive...")

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:

        members = archive.infolist()

        print(
            "Files in archive:",
            len(members),
        )

        for member in members:
            if member.is_dir():
                continue

            suffix = Path(
                member.filename
            ).suffix.lower()

            if suffix not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue

            image_id = normalize_image_id(
                member.filename
            )

            if image_id not in selected_ids:
                continue

            output_path = (
                IMAGES
                / f"{image_id}.jpg"
            )

            if output_path.exists():
                already_exists += 1
                continue

            with archive.open(
                member,
                "r",
            ) as source:
                with output_path.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(
                        source,
                        target,
                    )

            extracted += 1

    print(
        "Extracted selected:",
        extracted,
    )

    print(
        "Already existed:",
        already_exists,
    )

    print()
    print("Deleting archive...")

    archive_path.unlink()

    print("Archive deleted.")

    return extracted


def main():
    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            SELECTION_FILE
        )

    IMAGES.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_ids = load_selected_ids()

    print()
    print("=" * 60)
    print("CrowdHuman selected image downloader")
    print("=" * 60)

    print()
    print(
        "Selected IDs:",
        len(selected_ids),
    )

    before = {
        p.stem
        for p in IMAGES.glob("*.jpg")
    }

    print(
        "Already downloaded:",
        len(before & selected_ids),
    )

    for archive_name in ARCHIVES:
        process_archive(
            archive_name,
            selected_ids,
        )

    downloaded = {
        p.stem
        for p in IMAGES.glob("*.jpg")
    }

    matched = (
        downloaded
        & selected_ids
    )

    missing = (
        selected_ids
        - downloaded
    )

    extra = (
        downloaded
        - selected_ids
    )

    print()
    print("=" * 60)
    print("FINAL CHECK")
    print("=" * 60)

    print()
    print(
        "Selected:",
        len(selected_ids),
    )

    print(
        "Downloaded JPG:",
        len(downloaded),
    )

    print(
        "Matched:",
        len(matched),
    )

    print(
        "Missing:",
        len(missing),
    )

    print(
        "Extra:",
        len(extra),
    )

    if missing:
        print()
        print("First missing IDs:")

        for image_id in sorted(
            missing
        )[:20]:
            print(image_id)

    if extra:
        print()
        print("First extra IDs:")

        for image_id in sorted(
            extra
        )[:20]:
            print(image_id)

    if len(matched) != 2000:
        raise RuntimeError(
            "Not all selected CrowdHuman "
            "images were downloaded."
        )

    if missing:
        raise RuntimeError(
            "Missing selected images."
        )

    if extra:
        raise RuntimeError(
            "Unexpected extra images."
        )

    print()
    print(
        "All 2000 selected CrowdHuman "
        "images are ready."
    )


if __name__ == "__main__":
    main()