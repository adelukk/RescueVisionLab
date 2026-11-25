from pathlib import Path

import torch
from PIL import ImageDraw, ImageFont
from torchvision.ops import nms

from src.inference.predictor import predict


AVAILABLE_MODELS = [
    "YOLO",
    "Faster R-CNN",
    "RetinaNet",
]

MODEL_MAPPING = {
    "YOLO": "yolo26n",
    "Faster R-CNN": "fasterrcnn",
    "RetinaNet": "retinanet",
}

CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "final_checkpoints"
)


def detect_people(image, model_name, confidence, nms_iou):
    if image is None:
        raise ValueError("Сначала загрузите изображение.")

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Неизвестная модель: {model_name}")

    internal_name = MODEL_MAPPING[model_name]

    prediction = predict(
        image=image,
        model_name=internal_name,
        confidence=float(confidence),
        checkpoint_root=CHECKPOINT_ROOT,
    )
    boxes = prediction["boxes"]
    scores = prediction["scores"]

    if boxes:
        boxes_tensor = torch.tensor(
            boxes,
            dtype=torch.float32,
        )
        scores_tensor = torch.tensor(
            scores,
            dtype=torch.float32,
        )

        keep_indices = nms(
            boxes_tensor,
            scores_tensor,
            float(nms_iou),
        ).tolist()

        boxes = [boxes[index] for index in keep_indices]
        scores = [scores[index] for index in keep_indices]

    detections = []

    for box, score in zip(
        boxes,
        scores,
    ):
        x1, y1, x2, y2 = box
        detections.append(
            {
                "class": "person",
                "confidence": float(score),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
            }
        )

    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    image_width, image_height = result_image.size

    for detection in detections:
        x1 = max(0, min(int(detection["x1"]), image_width - 1))
        y1 = max(0, min(int(detection["y1"]), image_height - 1))
        x2 = max(0, min(int(detection["x2"]), image_width - 1))
        y2 = max(0, min(int(detection["y2"]), image_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        label = f"person {detection['confidence']:.2f}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_height - 8)

        draw.rectangle(
            [
                x1,
                label_y,
                min(x1 + text_width + 8, image_width - 1),
                min(label_y + text_height + 8, image_height - 1),
            ],
            fill="red",
        )
        draw.text(
            (x1 + 4, label_y + 4),
            label,
            fill="white",
            font=font,
        )

    message = (
        f"Модель: {model_name}\n"
        f"Обнаружено людей: {len(detections)}\n"
        f"Время инференса: {prediction['inference_ms']:.2f} мс\n"
        f"Confidence: {float(confidence):.2f}\n"
        f"NMS IoU: {float(nms_iou):.2f}"
    )

    return {
        "image": result_image,
        "detections": detections,
        "message": message,
    }
