from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def load_torchvision_metrics(eval_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(eval_dir.glob("*_*_metrics.csv")):
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            p = float(r.get("precision", 0.0))
            rec = float(r.get("recall", 0.0))
            rows.append({
                "model": str(r["model"]),
                "framework": "torchvision",
                "split": str(r["split"]),
                "checkpoint_epoch": int(r.get("checkpoint_epoch", -1)),
                "score_threshold": float(r.get("score_threshold", np.nan)),
                "iou_threshold": float(r.get("iou_threshold", np.nan)),
                "map50": float(r.get("map50", np.nan)),
                "map50_95": float(r.get("map50_95", np.nan)),
                "map75": float(r.get("map75", np.nan)),
                "precision": p,
                "recall": rec,
                "f1": compute_f1(p, rec),
                "tp": int(r.get("tp", 0)),
                "fp": int(r.get("fp", 0)),
                "fn": int(r.get("fn", 0)),
            })
    return pd.DataFrame(rows)


def plot_training_histories(logs_dir: Path, metrics_df: pd.DataFrame, out_path: Path) -> None:
    history_files = sorted(logs_dir.glob("*_history.csv"))
    if not history_files:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    test_df = metrics_df[metrics_df["split"] == "test"].set_index("model") if not metrics_df.empty else pd.DataFrame()

    for idx, h_path in enumerate(history_files[:4]):
        ax = axes[idx]
        model_name = h_path.stem.replace("_history", "")
        df = pd.read_csv(h_path)

        epochs = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"], marker="o", markersize=3, label="train_loss")
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"], marker="s", markersize=3, label="val_loss")

        if model_name in test_df.index and "val_loss" in df.columns:
            ckpt = int(test_df.loc[model_name, "checkpoint_epoch"])
            match = df[df["epoch"] == ckpt] if "epoch" in df.columns else pd.DataFrame()
            if not match.empty:
                val_at_ckpt = float(match["val_loss"].iloc[0])
                ax.scatter([ckpt], [val_at_ckpt], color="red", zorder=5, label=f"ckpt (ep {ckpt})")

        ax.set_title(f"{model_name} — динамика loss")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_test_comparison(all_df: pd.DataFrame, out_path: Path) -> None:
    test_df = all_df[all_df["split"] == "test"].copy()
    if test_df.empty:
        return

    test_df = test_df.sort_values("map50", ascending=False)
    models = test_df["model"].tolist()
    metric_cols = ["map50", "map50_95", "precision", "recall", "f1"]
    labels = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1"]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (col, lbl) in enumerate(zip(metric_cols, labels)):
        vals = [float(v) if pd.notna(v) else 0.0 for v in test_df[col]]
        ax.bar(x + (i - 2) * width, vals, width=width, label=lbl)

    ax.set_title("Сравнение моделей на тестовой выборке PDD (132 изображения)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, max(0.65, float(test_df[metric_cols].max().max()) + 0.1))
    ax.set_ylabel("Значение метрики")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнительная оценка моделей RescueVisionLab")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path(r"C:\Users\user\Downloads\RescueVisionLab_experiments"),
        help="Путь к распакованной папке экспериментов",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Папка для сохранения таблиц и графиков",
    )
    parser.add_argument("--yolo-split", type=str, default="test")
    parser.add_argument("--yolo-map50", type=float, default=None)
    parser.add_argument("--yolo-map50-95", type=float, default=None)
    parser.add_argument("--yolo-precision", type=float, default=None)
    parser.add_argument("--yolo-recall", type=float, default=None)
    parser.add_argument("--yolo-f1", type=float, default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    args = parser.parse_args()

    eval_dir = args.exp_dir / "evaluation"
    logs_dir = args.exp_dir / "training_logs"

    df = load_torchvision_metrics(eval_dir)

    if any(v is not None for v in [args.yolo_map50, args.yolo_map50_95, args.yolo_precision, args.yolo_recall, args.yolo_f1]):
        p = args.yolo_precision if args.yolo_precision is not None else np.nan
        r = args.yolo_recall if args.yolo_recall is not None else np.nan
        f1 = args.yolo_f1 if args.yolo_f1 is not None else (compute_f1(p, r) if pd.notna(p) and pd.notna(r) else np.nan)
        yolo_row = {
            "model": "yolo26n",
            "framework": "ultralytics",
            "split": args.yolo_split,
            "checkpoint_epoch": -1,
            "score_threshold": args.yolo_conf if args.yolo_conf is not None else np.nan,
            "iou_threshold": args.yolo_iou,
            "map50": args.yolo_map50 if args.yolo_map50 is not None else np.nan,
            "map50_95": args.yolo_map50_95 if args.yolo_map50_95 is not None else np.nan,
            "map75": np.nan,
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        df = pd.concat([df, pd.DataFrame([yolo_row])], ignore_index=True)

    metrics_out = args.out_dir / "metrics"
    plots_out = args.out_dir / "plots"
    metrics_out.mkdir(parents=True, exist_ok=True)
    plots_out.mkdir(parents=True, exist_ok=True)

    csv_out = metrics_out / "models_comparison.csv"
    json_out = metrics_out / "models_comparison.json"
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient="records", indent=2, force_ascii=False)

    plot_training_histories(logs_dir, df, plots_out / "training_losses.png")
    plot_test_comparison(df, plots_out / "models_test_comparison.png")

    print("Сохранено:", csv_out)
    print("Сохранено:", json_out)
    print("Сохранено:", plots_out / "training_losses.png")
    print("Сохранено:", plots_out / "models_test_comparison.png")
    print("\n--- Сводка по TEST ---")
    cols = ["model", "checkpoint_epoch", "score_threshold", "iou_threshold", "map50", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn"]
    print(df[df["split"] == "test"][cols].to_string(index=False))


if __name__ == "__main__":
    main()