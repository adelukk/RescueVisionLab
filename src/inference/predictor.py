
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.models.factory import create_model


import os

DEFAULT_CHECKPOINTS = Path(
    os.environ.get(
        "RESCUEVISION_CHECKPOINTS",
        r"C:\RescueVisionLab_Shared\checkpoints",
    )
)

MODEL_NAMES = (
    "yolo26n",
    "fasterrcnn",
    "retinanet",
    "ssd300",
    "fcos",
)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model(model_name, checkpoint_root=None):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")

    root = Path(checkpoint_root or DEFAULT_CHECKPOINTS)
    device = get_device()

    if model_name == "yolo26n":
        path = root / "yolo26n" / "best.pt"

        if not path.is_file():
            raise FileNotFoundError(path)

        return YOLO(str(path))

    path = root / model_name / "best.pth"

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["model_name"] != model_name:
        raise ValueError("Checkpoint model mismatch")

    model = create_model(model_name)

    if model_name != "ssd300":
        model.transform.min_size = (
            checkpoint["min_size"],
        )
        model.transform.max_size = checkpoint["max_size"]

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def predict(
    image,
    model_name="yolo26n",
    confidence=0.25,
    checkpoint_root=None,
):
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image or file path")

    image = image.convert("RGB")
    model = load_model(model_name, checkpoint_root)
    device = get_device()

    start = perf_counter()

    if model_name == "yolo26n":
        result = model.predict(
            source=np.asarray(image),
            conf=confidence,
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )[0]

        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        labels = result.boxes.cls.int().cpu().tolist()

    else:
        array = np.asarray(image).copy()
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(device)
        )

        with torch.inference_mode():
            result = model([tensor])[0]

        # Torchvision: person = 1
        mask = (
            (result["scores"] >= confidence)
            & (result["labels"] == 1)
        )

        boxes = result["boxes"][mask].cpu().tolist()
        scores = result["scores"][mask].cpu().tolist()

        # Unified output: person = 0
        labels = [0] * len(scores)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (perf_counter() - start) * 1000

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "inference_ms": elapsed_ms,
    }
