from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HISTORY_DIR = Path("results_final/training_history")
OUTPUT_DIR = Path("results_final/plots")


def plot_torchvision_history(model_name, filename, best_epoch):
    data = pd.read_csv(HISTORY_DIR / filename)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        data["epoch"],
        data["train_loss"],
        marker="o",
        label="Train loss",
    )
    ax.plot(
        data["epoch"],
        data["val_loss"],
        marker="o",
        label="Validation loss",
    )

    best_row = data.loc[data["epoch"] == best_epoch].iloc[0]

    ax.scatter(
        [best_epoch],
        [best_row["val_loss"]],
        s=90,
        zorder=5,
        label=f"Best epoch: {best_epoch}",
    )

    ax.set_title(f"{model_name}: training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_loss.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("Saved:", output)


def plot_yolo_history():
    data = pd.read_csv(HISTORY_DIR / "yolo26n_history.csv")
    data.columns = data.columns.str.strip()

    best_epoch = 28

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        data["epoch"],
        data["metrics/mAP50(B)"],
        marker="o",
        label="mAP@0.5",
    )
    ax.plot(
        data["epoch"],
        data["metrics/mAP50-95(B)"],
        marker="o",
        label="mAP@0.5:0.95",
    )

    best_row = data.loc[data["epoch"] == best_epoch].iloc[0]

    ax.scatter(
        [best_epoch],
        [best_row["metrics/mAP50-95(B)"]],
        s=90,
        zorder=5,
        label=f"Best epoch: {best_epoch}",
    )

    ax.set_title("YOLO26n: validation metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = OUTPUT_DIR / "yolo26n_map.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("Saved:", output)


def plot_yolo_losses():
    data = pd.read_csv(HISTORY_DIR / "yolo26n_history.csv")
    data.columns = data.columns.str.strip()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        data["epoch"],
        data["train/box_loss"],
        label="Train box loss",
    )
    ax.plot(
        data["epoch"],
        data["val/box_loss"],
        label="Validation box loss",
    )

    ax.set_title("YOLO26n: box loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Box loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output = OUTPUT_DIR / "yolo26n_box_loss.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("Saved:", output)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_torchvision_history(
        "Faster R-CNN",
        "fasterrcnn_history.csv",
        best_epoch=7,
    )

    plot_torchvision_history(
        "RetinaNet",
        "retinanet_history.csv",
        best_epoch=10,
    )

    plot_yolo_history()
    plot_yolo_losses()

    print()
    print("All final training plots generated.")


if __name__ == "__main__":
    main()