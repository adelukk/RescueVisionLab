import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    fcos_resnet50_fpn,
)


MODEL_NAMES = [
    "fasterrcnn",
    "retinanet",
    "ssd300",
    "fcos",
]


def create_model(name: str):
    """
    Создаёт модель обнаружения объектов
    со случайными начальными весами.

    В torchvision:
        0 = background
        1 = person
    """

    if name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )

    elif name == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )

    elif name == "ssd300":
        model = ssd300_vgg16(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )

    elif name == "fcos":
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=2,
        )

    else:
        raise ValueError(
            f"Неизвестная модель: {name}. "
            f"Доступны: {MODEL_NAMES}"
        )

    return model


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )