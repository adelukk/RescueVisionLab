import os
from pathlib import Path

from src.inference.predictor import MODEL_NAMES, predict


IMAGE_PATH = Path(
    os.environ.get(
        "RESCUEVISION_TEST_IMAGE",
        r"C:\RescueVisionLab_Shared\dataset\pdd"
        r"\images\test\02_0005.png",
    )
)

CHECKPOINT_ROOT = Path(
    os.environ.get(
        "RESCUEVISION_CHECKPOINTS",
        r"C:\RescueVisionLab_Shared\checkpoints",
    )
)


def main():
    for model_name in MODEL_NAMES:
        result = predict(
            IMAGE_PATH,
            model_name=model_name,
            confidence=0.25,
            checkpoint_root=CHECKPOINT_ROOT,
        )

        assert isinstance(result, dict)
        assert set(result) == {
            "boxes", "scores", "labels", "inference_ms"
        }

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        assert len(boxes) == len(scores) == len(labels)
        assert result["inference_ms"] >= 0

        for box, score, label in zip(
            boxes, scores, labels
        ):
            assert len(box) == 4
            assert box[0] <= box[2]
            assert box[1] <= box[3]
            assert 0 <= score <= 1
            assert label == 0

        print(
            f"PASS: {model_name}, "
            f"detections={len(boxes)}"
        )

    print("All contract checks passed.")


if __name__ == "__main__":
    main()
