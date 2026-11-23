
from pathlib import Path
import gc

import torch

from src.inference.predictor import predict, load_model


TEST_DIR = Path(
    r"C:\RescueVisionLab_Shared\dataset\pdd\images\test"
)

MODEL_NAMES = [
    "yolo26n",
    "fasterrcnn",
    "retinanet",
    "ssd300",
    "fcos",
]


def main():
    image_path = next(TEST_DIR.glob("*.png"))

    print("Test image:", image_path)

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 50}")
        print("MODEL:", model_name)
        print("=" * 50)

        try:
            result = predict(
                image=image_path,
                model_name=model_name,
                confidence=0.25,
            )

            print("Detected persons:", len(result["boxes"]))
            print("Boxes:", result["boxes"])
            print("Scores:", result["scores"])
            print(
                "Inference time (ms):",
                round(result["inference_ms"], 2),
            )

        except Exception as error:
            print("ERROR:", repr(error))

        finally:
            # Release cached models between tests
            load_model.cache_clear()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll model checks finished.")


if __name__ == "__main__":
    main()
