from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results_final")
PLOTS_DIR = RESULTS_DIR / "plots"
OUTPUT_CSV = RESULTS_DIR / "final_model_comparison.csv"


def main():
    faster = pd.read_csv(
        RESULTS_DIR / "fasterrcnn_test_metrics.csv"
    ).iloc[0]

    retina = pd.read_csv(
        RESULTS_DIR / "retinanet_test_metrics.csv"
    ).iloc[0]

    yolo = pd.read_csv(
        RESULTS_DIR / "yolo26n_test_threshold_metrics.csv"
    ).iloc[0]

    speed = pd.read_csv(
        RESULTS_DIR / "speed_benchmark.csv"
    ).set_index("model")

    rows = [
        {
            "model": "YOLO26n",
            "best_epoch": 28,
            "map50": 0.713,
            "map50_95": 0.403,
            "precision_at_0.5": yolo["precision"],
            "recall_at_0.5": yolo["recall"],
            "tp": int(yolo["tp"]),
            "fp": int(yolo["fp"]),
            "fn": int(yolo["fn"]),
            "mean_ms": speed.loc["yolo26n", "mean_ms"],
            "fps": speed.loc["yolo26n", "fps"],
        },
        {
            "model": "Faster R-CNN",
            "best_epoch": int(faster["checkpoint_epoch"]),
            "map50": faster["map50"],
            "map50_95": faster["map50_95"],
            "precision_at_0.5": faster["precision"],
            "recall_at_0.5": faster["recall"],
            "tp": int(faster["tp"]),
            "fp": int(faster["fp"]),
            "fn": int(faster["fn"]),
            "mean_ms": speed.loc["fasterrcnn", "mean_ms"],
            "fps": speed.loc["fasterrcnn", "fps"],
        },
        {
            "model": "RetinaNet",
            "best_epoch": int(retina["checkpoint_epoch"]),
            "map50": retina["map50"],
            "map50_95": retina["map50_95"],
            "precision_at_0.5": retina["precision"],
            "recall_at_0.5": retina["recall"],
            "tp": int(retina["tp"]),
            "fp": int(retina["fp"]),
            "fn": int(retina["fn"]),
            "mean_ms": speed.loc["retinanet", "mean_ms"],
            "fps": speed.loc["retinanet", "fps"],
        },
    ]

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # mAP comparison
    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(comparison))
    width = 0.35

    ax.bar(
        [value - width / 2 for value in x],
        comparison["map50"],
        width,
        label="mAP@0.5",
    )

    ax.bar(
        [value + width / 2 for value in x],
        comparison["map50_95"],
        width,
        label="mAP@0.5:0.95",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mAP")
    ax.set_title("Detection accuracy on PDD test")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "model_map_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    # Precision / Recall comparison
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        [value - width / 2 for value in x],
        comparison["precision_at_0.5"],
        width,
        label="Precision",
    )

    ax.bar(
        [value + width / 2 for value in x],
        comparison["recall_at_0.5"],
        width,
        label="Recall",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Metric value")
    ax.set_title(
        "Precision and Recall at confidence = 0.5"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "precision_recall_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    # Speed comparison
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        comparison["model"],
        comparison["fps"],
    )

    ax.set_ylabel("FPS")
    ax.set_title(
        "Inference speed on NVIDIA RTX 4050 Laptop GPU"
    )
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, comparison["fps"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "speed_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    print("Saved:", OUTPUT_CSV)
    print(
        "Saved:",
        PLOTS_DIR / "model_map_comparison.png",
    )
    print(
        "Saved:",
        PLOTS_DIR / "precision_recall_comparison.png",
    )
    print(
        "Saved:",
        PLOTS_DIR / "speed_comparison.png",
    )

    print()
    print("=== FINAL MODEL COMPARISON ===")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()