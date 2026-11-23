
from pathlib import Path
import gc

import torch
from PIL import Image, ImageDraw, ImageFont

from src.inference.predictor import predict, load_model


DATASET_DIR = Path(
    r"C:\RescueVisionLab_Shared\dataset\pdd\images\test"
)

OUTPUT_DIR = Path("results/visualizations")

MODELS = [
    "yolo26n",
    "fasterrcnn",
    "retinanet",
    "ssd300",
    "fcos",
]


def draw_predictions(image, result):
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)

    for box, score in zip(
        result["boxes"], result["scores"]
    ):
        x1, y1, x2, y2 = box

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"person {score:.2f}",
            fill="red",
        )

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = next(DATASET_DIR.glob("*.png"))
    image = Image.open(image_path).convert("RGB")

    print("Image:", image_path)

    for model_name in MODELS:
        print("Processing:", model_name)

        try:
            result = predict(
                image=image,
                model_name=model_name,
                confidence=0.25,
            )

            visualization = draw_predictions(
                image, result
            )

            output_path = (
                OUTPUT_DIR / f"{model_name}_result.png"
            )

            visualization.save(output_path)

            print(
                f"Saved: {output_path}, "
                f"detections: {len(result['boxes'])}"
            )

        finally:
            load_model.cache_clear()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()
