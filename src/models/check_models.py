import gc
import torch

from src.models.factory import (
    MODEL_NAMES,
    create_model,
)


for name in MODEL_NAMES:
    print(f"\nСоздание модели: {name}")

    model = create_model(name)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Количество параметров: {parameters:,}"
    )

    print("Модель успешно создана")

    del model
    gc.collect()

print("\nВсе четыре архитектуры созданы")