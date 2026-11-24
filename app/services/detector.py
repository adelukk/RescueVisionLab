from pathlib import Path

from src.inference.predictor import predict

AVAILABLE_MODELS = [
    "YOLO",
    "Faster R-CNN",
    "RetinaNet",
    "SSD300",
    "FCOS",
]

MODEL_MAPPING = {
    "YOLO": "yolo26n",
    "Faster R-CNN": "fasterrcnn",
    "RetinaNet": "retinanet",
    "SSD300": "ssd300",
    "FCOS": "fcos",
}

# Фактическая папка с весами в проекте
CHECKPOINT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / "checkpoints"
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

    detections = []

    for box, score in zip(
        prediction["boxes"],
        prediction["scores"],
    ):
        x1, y1, x2, y2 = box

        detections.append({
            "class": "person",
            "confidence": float(score),
            "x1": round(float(x1), 2),
            "y1": round(float(y1), 2),
            "x2": round(float(x2), 2),
            "y2": round(float(y2), 2),
        })

    result_image = image.copy()

    return {
        "image": result_image,
        "detections": detections,
        "message": (
            f"Модель: {model_name}\n"
            f"Обнаружено людей: {len(detections)}\n"
            f"Время инференса: "
            f"{prediction['inference_ms']:.2f} мс\n"
            f"Confidence: {confidence}\n"
            f"NMS IoU: {nms_iou}"
        ),
    }