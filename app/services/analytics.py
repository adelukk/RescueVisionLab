from io import BytesIO
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from PIL import Image


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results_final"
HISTORY_FILES = {
    "YOLO26n": "yolo26n_history.csv",
    "Faster R-CNN": "fasterrcnn_history.csv",
    "RetinaNet": "retinanet_history.csv",
}
COMPARISON_HEADERS = {
    "model": "Модель",
    "best_epoch": "Лучшая эпоха",
    "map50": "mAP50",
    "map50_95": "mAP50–95",
    "precision_at_0.5": "Precision (confidence = 0,5)",
    "recall_at_0.5": "Recall (confidence = 0,5)",
    "tp": "TP",
    "fp": "FP",
    "fn": "FN",
    "mean_ms": "Среднее время, мс",
    "fps": "FPS",
}


def read_results(path, columns):
    if not path.is_file():
        raise ValueError(f"Файл не найден: {path.relative_to(RESULTS_DIR)}")
    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"В {path.name} отсутствуют столбцы: {', '.join(sorted(missing))}.")
    data = data.loc[:, list(columns)].copy()
    for column in data.columns.drop("model", errors="ignore"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.replace([float("inf"), -float("inf")], float("nan"))


def figure_to_image(figure):
    with BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=130)
        buffer.seek(0)
        with Image.open(buffer) as image:
            return image.convert("RGB")


def plot_history(axis, data, series, title, ylabel, limits=None):
    for column, label in series.items():
        axis.plot(data["epoch"], data[column], marker="o", markersize=3, label=label)
    axis.set(title=title, xlabel="Эпоха", ylabel=ylabel, ylim=limits)
    axis.grid(alpha=0.25)
    axis.legend()


def load_training_history(model):
    loss = {"train_loss": "Обучение", "val_loss": "Валидация"}
    map_series = {}
    loss_label = "Потери"
    if model == "YOLO26n":
        loss = {"train/box_loss": "Обучение", "val/box_loss": "Валидация"}
        map_series = {"metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50–95"}
        loss_label = "Потери локализации (box loss)"

    try:
        path = RESULTS_DIR / "training_history" / HISTORY_FILES[model]
        data = read_results(path, ["epoch", *loss, *map_series])
        data = data.dropna(subset=["epoch", *loss, *map_series]).sort_values("epoch")
        if data.empty:
            raise ValueError(f"В {path.name} нет заполненных данных для графика.")

        figure = Figure(figsize=(9, 7 if map_series else 4), layout="constrained")
        axes = figure.subplots(2 if map_series else 1, 1, squeeze=False)
        plot_history(axes[0, 0], data, loss, f"{model} · {loss_label}", loss_label)
        if map_series:
            plot_history(axes[1, 0], data, map_series, "Динамика mAP на валидации", "mAP", (0, 1))
        return figure_to_image(figure), f"Эпох на графике: {len(data)}."
    except (OSError, ValueError, KeyError) as error:
        return None, f"Не удалось загрузить историю обучения. {error}"


def load_model_comparison():
    empty_table = pd.DataFrame(columns=list(COMPARISON_HEADERS.values()))
    try:
        data = read_results(RESULTS_DIR / "final_model_comparison.csv", COMPARISON_HEADERS)
        data["model"] = data["model"].str.strip()
        data = data[data["model"].isin(HISTORY_FILES)]
        if set(data["model"]) != set(HISTORY_FILES) or data["model"].duplicated().any():
            raise ValueError("Ожидается по одной строке для YOLO26n, Faster R-CNN и RetinaNet.")
        if data.isna().any().any():
            raise ValueError("В итоговом сравнении есть пустые или некорректные значения.")
        data = data.set_index("model").loc[list(HISTORY_FILES)]

        figure = Figure(figsize=(12, 4), layout="constrained")
        axes = figure.subplots(1, 3)
        panels = [
            (["map50", "map50_95"], ["mAP50", "mAP50–95"], "Средняя точность (mAP)", (0, 1)),
            (
                ["precision_at_0.5", "recall_at_0.5"],
                ["Precision", "Recall"],
                "Precision и Recall\nconfidence = 0,5",
                (0, 1),
            ),
            (["fps"], ["FPS"], "Скорость, кадров/с", None),
        ]
        for axis, (columns, labels, title, limits) in zip(axes, panels):
            data[columns].set_axis(labels, axis=1).plot.bar(
                ax=axis, rot=0, color=["#2463a8", "#d67a26"], legend=len(columns) > 1
            )
            axis.set(title=title, xlabel="", ylim=limits)
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.25)

        table = data.reset_index().rename(columns=COMPARISON_HEADERS).round(4)
        return figure_to_image(figure), table, ""
    except (OSError, ValueError) as error:
        return None, empty_table, f"Не удалось загрузить сравнение моделей. {error}"
