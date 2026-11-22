import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    args = parser.parse_args()

    # Создаём модель с нуля.
    model = YOLO("yolo26n.yaml")

    model.train(
        data="configs/pdd.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=2,
        seed=42,
        project="runs/rescuevision",
        name="yolo26n",
        exist_ok=True,
        pretrained=False,
    )


if __name__ == "__main__":
    main()