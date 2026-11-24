
import gc
import os
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.models.factory import create_model


MODEL_NAMES = ("yolo26n", "fasterrcnn", "retinanet", "ssd300", "fcos")

DEFAULT_CHECKPOINTS = Path(
    os.environ.get(
        "RESCUEVISION_CHECKPOINTS",
        r"C:\RescueVisionLab_Shared\checkpoints",
    )
)

_cached_model = None
_cached_key = None
_model_lock = threading.RLock()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_model_cache():
    global _cached_model, _cached_key

    with _model_lock:
        _cached_model = None
        _cached_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_model_uncached(model_name, root):
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
        path, map_location="cpu", weights_only=True
    )

    if checkpoint["model_name"] != model_name:
        raise ValueError("Checkpoint model mismatch")

    model = create_model(model_name)

    if model_name != "ssd300":
        model.transform.min_size = (checkpoint["min_size"],)
        model.transform.max_size = checkpoint["max_size"]

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def load_model(model_name, checkpoint_root=None):
    global _cached_model, _cached_key

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")

    root = Path(checkpoint_root or DEFAULT_CHECKPOINTS).resolve()
    key = (model_name, str(root), get_device())

    with _model_lock:
        if _cached_model is not None and _cached_key == key:
            return _cached_model

        clear_model_cache()

        model = _load_model_uncached(model_name, root)
        _cached_model = model
        _cached_key = key
        return model


def predict(image, model_name="yolo26n",
            confidence=0.25, checkpoint_root=None):

    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image or file path")

    image = image.convert("RGB")
    device = get_device()

    # Keep model switching and inference serialized.
    with _model_lock:
        model = load_model(model_name, checkpoint_root)

        if device == "cuda":
            torch.cuda.synchronize()
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

            mask = (
                (result["scores"] >= confidence)
                & (result["labels"] == 1)
            )

            boxes = result["boxes"][mask].cpu().tolist()
            scores = result["scores"][mask].cpu().tolist()
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
