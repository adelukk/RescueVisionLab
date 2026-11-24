from pathlib import Path
import json
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "combined_sources"
    / "crowdhuman"
)

METADATA = ROOT / "metadata"

FILES = {
    "train": METADATA / "annotation_train.odgt",
    "val": METADATA / "annotation_val.odgt",
}


def is_ignored(box):
    extra = box.get("extra", {})
    return int(extra.get("ignore", 0)) == 1


def valid_fbox(box):
    fbox = box.get("fbox")

    if not isinstance(fbox, list) or len(fbox) != 4:
        return False

    x, y, w, h = fbox

    return (
        w > 0
        and h > 0
    )


def analyze(path):
    image_count = 0
    total_boxes = 0
    valid_people = 0
    ignored = 0
    invalid_fbox = 0
    people_per_image = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            image_count += 1
            people = 0

            for box in record.get("gtboxes", []):
                total_boxes += 1

                if is_ignored(box):
                    ignored += 1
                    continue

                if not valid_fbox(box):
                    invalid_fbox += 1
                    continue

                valid_people += 1
                people += 1

            people_per_image.append(people)

    zero = sum(x == 0 for x in people_per_image)
    one = sum(x == 1 for x in people_per_image)
    two_to_five = sum(2 <= x <= 5 for x in people_per_image)
    six_to_ten = sum(6 <= x <= 10 for x in people_per_image)
    eleven_to_twenty = sum(11 <= x <= 20 for x in people_per_image)
    over_twenty = sum(x > 20 for x in people_per_image)

    return {
        "images": image_count,
        "all_gtboxes": total_boxes,
        "valid_people": valid_people,
        "ignored": ignored,
        "invalid_fbox": invalid_fbox,
        "zero": zero,
        "one": one,
        "two_to_five": two_to_five,
        "six_to_ten": six_to_ten,
        "eleven_to_twenty": eleven_to_twenty,
        "over_twenty": over_twenty,
        "mean": statistics.mean(people_per_image),
        "median": statistics.median(people_per_image),
        "maximum": max(people_per_image),
    }


def print_stats(name, stats):
    print()
    print("=" * 50)
    print(name.upper())
    print("=" * 50)

    print("Images:", stats["images"])
    print("All gtboxes:", stats["all_gtboxes"])
    print("Valid people:", stats["valid_people"])
    print("Ignored:", stats["ignored"])
    print("Invalid fbox:", stats["invalid_fbox"])

    print()
    print("PEOPLE PER IMAGE")
    print("0:", stats["zero"])
    print("1:", stats["one"])
    print("2-5:", stats["two_to_five"])
    print("6-10:", stats["six_to_ten"])
    print("11-20:", stats["eleven_to_twenty"])
    print(">20:", stats["over_twenty"])

    print()
    print("Mean:", round(stats["mean"], 2))
    print("Median:", stats["median"])
    print("Maximum:", stats["maximum"])


def main():
    results = {}

    for split, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"Reading {path.name}...")
        results[split] = analyze(path)

    for split, stats in results.items():
        print_stats(split, stats)

    print()
    print("=" * 50)
    print("TOTAL")
    print("=" * 50)

    print(
        "Images:",
        sum(x["images"] for x in results.values())
    )

    print(
        "Valid people:",
        sum(x["valid_people"] for x in results.values())
    )

    print(
        "Ignored:",
        sum(x["ignored"] for x in results.values())
    )


if __name__ == "__main__":
    main()