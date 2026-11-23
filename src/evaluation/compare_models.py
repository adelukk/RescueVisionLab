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


def load_yolo_history_summary(logs_dir: Path) -> pd.DataFrame:
    y_path = logs_dir / "yolo26n_history.csv"
    if not y_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(y_path)
    if df.empty:
        return pd.DataFrame()

    best_idx = int(df["metrics/mAP50(B)"].idxmax()) if "metrics/mAP50(B)" in df.columns else len(df) - 1
    best_row = df.loc[best_idx]
    p = float(best_row.get("metrics/precision(B)", np.nan))
    rec = float(best_row.get("metrics/recall(B)", np.nan))
    f1 = compute_f1(p, rec) if pd.notna(p) and pd.notna(rec) else np.nan

    return pd.DataFrame([{
        "model": "yolo26n",
        "framework": "ultralytics",
        "split": "val_ultralytics",
        "checkpoint_epoch": int(best_row.get("epoch", best_idx + 1)),
        "score_threshold": np.nan,
        "iou_threshold": np.nan,
        "map50": float(best_row.get("metrics/mAP50(B)", np.nan)),
        "map50_95": float(best_row.get("metrics/mAP50-95(B)", np.nan)),
        "map75": np.nan,
        "precision": p,
        "recall": rec,
        "f1": f1,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }])


def plot_training_histories(logs_dir: Path, metrics_df: pd.DataFrame, out_path: Path) -> None:
    tv_files = sorted(
        p for p in logs_dir.glob("*_history.csv") if not p.stem.startswith("yolo26n")
    )
    y_path = logs_dir / "yolo26n_history.csv"

    total_panels = len(tv_files) + (1 if y_path.exists() else 0)
    if total_panels == 0:
        return

    cols = 3
    rows_n = int(np.ceil(total_panels / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, 4.5 * rows_n))
    axes = np.atleast_1d(axes).ravel()

    test_df = metrics_df[metrics_df["split"] == "test"].set_index("model") if not metrics_df.empty else pd.DataFrame()
    panel_idx = 0

    for h_path in tv_files:
        ax = axes[panel_idx]
        panel_idx += 1
        model_name = h_path.stem.replace("_history", "")
        df = pd.read_csv(h_path)

        epochs = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
        if "train_loss" in df.columns:
            ax.plot(epochs, df["train_loss"], marker="o", markersize=2.5, label="train_loss")
        if "val_loss" in df.columns:
            ax.plot(epochs, df["val_loss"], marker="s", markersize=2.5, label="val_loss")

        if model_name in test_df.index and "val_loss" in df.columns:
            ckpt = int(test_df.loc[model_name, "checkpoint_epoch"])
            match = df[df["epoch"] == ckpt] if "epoch" in df.columns else pd.DataFrame()
            if not match.empty:
                val_at_ckpt = float(match["val_loss"].iloc[0])
                ax.scatter([ckpt], [val_at_ckpt], color="red", zorder=5, label=f"ckpt (ep {ckpt})")

        ax.set_title(f"{model_name} — train/val loss")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    if y_path.exists():
        ax = axes[panel_idx]
        panel_idx += 1
        ydf = pd.read_csv(y_path)
        epochs = ydf["epoch"] if "epoch" in ydf.columns else np.arange(1, len(ydf) + 1)
        train_cols = [c for c in ["train/box_loss", "train/cls_loss", "train/l1_loss"] if c in ydf.columns]
        val_cols = [c for c in ["val/box_loss", "val/cls_loss", "val/l1_loss"] if c in ydf.columns]
        if train_cols:
            ax.plot(epochs, ydf[train_cols].sum(axis=1), label="train sum loss")
        if val_cols:
            ax.plot(epochs, ydf[val_cols].sum(axis=1), label="val sum loss")
        ax.set_title("yolo26n — сумма потерь train/val")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(panel_idx, len(axes)):
        axes[j].set_visible(False)

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

    ax.set_title("Torchvision-модели на тесте (score=0.25, IoU=0.5; YOLO26n — отдельно в логах)")
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
        default=Path.home() / "Downloads" / "RescueVisionLab_For_Arina",
        help="Путь к распакованной папке экспериментов",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Папка для сохранения таблиц и графиков",
    )
    args = parser.parse_args()

    eval_dir = args.exp_dir / "evaluation"
    logs_dir = args.exp_dir / "training_logs"

    tv_df = load_torchvision_metrics(eval_dir)
    yolo_df = load_yolo_history_summary(logs_dir)
    df = pd.concat([tv_df, yolo_df], ignore_index=True)

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
    print("\n--- Сводка по всем записям ---")
    cols = ["model", "split", "checkpoint_epoch", "score_threshold", "iou_threshold", "map50", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()