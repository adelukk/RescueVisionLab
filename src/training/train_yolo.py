import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="yolo26n_final",
    )

    args = parser.parse_args()

    print("Starting final YOLO26n training from scratch.")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Run name: {args.name}")

    # Создаём архитектуру YOLO26n без предобученных весов.
    model = YOLO("yolo26n.yaml")

    model.train(
        data="datasets/processed/combined_yolo/data.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        workers=2,
        seed=42,
        project="runs/rescuevision",
        name=args.name,
        exist_ok=False,
        pretrained=False,
    )


if __name__ == "__main__":
    main()