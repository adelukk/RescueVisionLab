import argparse
import gc
import tempfile
from pathlib import Path

import torch

from src.models.factory import (
    MODEL_NAMES,
    create_model,
    get_device,
)


def smoke_test(name, device):
    print(f"\n=== Проверка {name} ===")
    print("Устройство:", device)

    model = create_model(name).to(device)

    # Искусственное изображение RGB:
    # C=3, H=320, W=320
    image = torch.rand(
        3, 320, 320,
        dtype=torch.float32,
        device=device,
    )

    # Истинная рамка в абсолютных пикселях.
    target = {
        "boxes": torch.tensor(
            [[70.0, 60.0, 230.0, 270.0]],
            dtype=torch.float32,
            device=device,
        ),
        "labels": torch.tensor(
            [1],
            dtype=torch.int64,
            device=device,
        ),
    }

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
    )

    # 1. Режим обучения
    model.train()

    # 2. Обнуляем градиенты
    optimizer.zero_grad(set_to_none=True)

    # 3. Прямой проход
    losses = model([image], [target])

    # 4. Суммарная ошибка
    total_loss = sum(losses.values())

    print("Потери:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    print("Total loss:", total_loss.item())

    if not torch.isfinite(total_loss):
        raise RuntimeError("Loss не является конечным числом")

    # 5. Обратное распространение
    total_loss.backward()

    # 6. Проверяем наличие градиентов
    has_gradients = any(
        p.grad is not None
        and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )

    if not has_gradients:
        raise RuntimeError("Градиенты не получены")

    # 7. Обновляем веса
    optimizer.step()

    print("Один шаг обучения выполнен")

    # 8. Сохраняем веса во временный файл
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}_smoke.pth"

        torch.save(model.state_dict(), path)

        print("Веса сохранены")

        # 9. Проверяем повторную загрузку
        state = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)

        print("Веса успешно загружены")

    # 10. Проверяем инференс
    model.eval()

    with torch.inference_mode():
        predictions = model([image])

    prediction = predictions[0]

    print("Формат предсказаний:")
    print("Boxes:", prediction["boxes"].shape)
    print("Scores:", prediction["scores"].shape)
    print("Labels:", prediction["labels"].shape)

    del model, optimizer, losses, total_loss
    del image, target, predictions, prediction

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Проверка {name} завершена")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    device = (
        get_device()
        if args.device == "auto"
        else torch.device(args.device)
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")

    smoke_test(args.model, device)


if __name__ == "__main__":
    main()