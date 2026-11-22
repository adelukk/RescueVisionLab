from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.yaml")

    parameters = sum(
        p.numel()
        for p in model.model.parameters()
    )

    print("Архитектура: YOLO26n")
    print("Параметры:", f"{parameters:,}")
    print("Модель создана из YAML")
    print("Предобученные веса не загружались")


if __name__ == "__main__":
    main()