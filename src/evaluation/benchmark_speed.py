import csv
import statistics
from pathlib import Path

from PIL import Image

from src.data.detection_dataset import PDDDataset
from src.inference.predictor import (
    MODEL_NAMES,
    clear_model_cache,
    get_device,
    predict,
)


CHECKPOINT_ROOT = Path("final_checkpoints")
OUTPUT_FILE = Path("results_final/speed_benchmark.csv")

WARMUP_IMAGES = 5
CONFIDENCE = 0.25


def tensor_to_pil(tensor):
    array = (
        tensor.mul(255)
        .clamp(0, 255)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def main():
    dataset = PDDDataset(split="test")

    print("Device:", get_device())
    print("Test images:", len(dataset))
    print("Models:", ", ".join(MODEL_NAMES))
    print("Warm-up images:", WARMUP_IMAGES)
    print("Confidence:", CONFIDENCE)

    images = [
        tensor_to_pil(dataset[index][0])
        for index in range(len(dataset))
    ]

    rows = []

    for model_name in MODEL_NAMES:
        print()
        print(f"=== {model_name} ===")

        clear_model_cache()

        # GPU/model warm-up. These measurements are discarded.
        for image in images[:WARMUP_IMAGES]:
            predict(
                image,
                model_name=model_name,
                confidence=CONFIDENCE,
                checkpoint_root=CHECKPOINT_ROOT,
            )

        print("Warm-up completed.")

        times = []

        for index, image in enumerate(images, 1):
            result = predict(
                image,
                model_name=model_name,
                confidence=CONFIDENCE,
                checkpoint_root=CHECKPOINT_ROOT,
            )

            times.append(result["inference_ms"])

            if index % 25 == 0 or index == len(images):
                print(
                    f"Benchmarked {index}/{len(images)}",
                    flush=True,
                )

        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        min_ms = min(times)
        max_ms = max(times)
        fps = 1000.0 / mean_ms

        row = {
            "model": model_name,
            "images": len(times),
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
            "fps": fps,
        }

        rows.append(row)

        print(f"Mean: {mean_ms:.2f} ms/image")
        print(f"Median: {median_ms:.2f} ms/image")
        print(f"Min: {min_ms:.2f} ms/image")
        print(f"Max: {max_ms:.2f} ms/image")
        print(f"FPS: {fps:.2f}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== FINAL SPEED COMPARISON ===")

    for row in rows:
        print(
            f'{row["model"]}: '
            f'{row["mean_ms"]:.2f} ms/image, '
            f'{row["fps"]:.2f} FPS'
        )

    print()
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()