from PIL import Image

AVAILABLE_MODELS = [
    "YOLO",
    "Faster R-CNN",
    "RetinaNet",
    "SSD300",
    "FCOS",
]

def detect_people(image, model_name, confidence, nms_iou):
    """
    Временная заглушка для будущего подключения обученных моделей.

    Сейчас реальные веса ещё не подключены.
    Поэтому функция не имитирует обнаружение людей.
    """
    if image is None:
        raise ValueError("Сначала загрузите изображение.")

    return {
        "image": image.copy(),
        "detections": [],
        "message": (
            f"Модель: {model_name}\n"
            "Обнаружение пока недоступно: обученные веса ещё не подключены.\n"
            f"Confidence: {confidence}\n"
            f"NMS IoU: {nms_iou}"
        ),
    }