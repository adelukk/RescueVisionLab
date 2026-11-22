import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO

def figure_to_image(fig):
    """Преобразует график Matplotlib в изображение PIL."""
    buffer = BytesIO()

    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)

    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def load_training_history(file_path):
    """Загружает историю обучения из CSV."""
    if file_path is None:
        return None, "Загрузите CSV с историей обучения."

    try:
        data = pd.read_csv(file_path)

        required = {"epoch", "train_loss", "val_loss"}

        if not required.issubset(data.columns):
            missing = required - set(data.columns)
            return None, (
                "В файле отсутствуют необходимые столбцы: "
                + ", ".join(sorted(missing))
            )

        data = data.dropna(subset=["epoch", "train_loss", "val_loss"])

        if data.empty:
            return None, "В CSV нет заполненных данных для построения графика."

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            data["epoch"],
            data["train_loss"],
            label="Train Loss",
            marker="o"
        )

        ax.plot(
            data["epoch"],
            data["val_loss"],
            label="Validation Loss",
            marker="o"
        )

        ax.set_title("История обучения модели")
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

        return figure_to_image(fig), (
            f"График построен. Количество эпох: {len(data)}."
        )

    except Exception as error:
        return None, f"Ошибка чтения истории обучения: {error}"

def load_model_comparison(file_path):
    """Загружает и визуализирует результаты сравнения моделей."""
    if file_path is None:
        return None, None, "Загрузите CSV с результатами оценки моделей."

    try:
        data = pd.read_csv(file_path)

        required = {
            "model",
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95"
        }

        if not required.issubset(data.columns):
            missing = required - set(data.columns)

            return (
                None,
                None,
                "В файле отсутствуют столбцы: "
                + ", ".join(sorted(missing))
            )

        data = data.dropna(subset=list(required))

        if data.empty:
            return None, None, "В CSV нет заполненных результатов."

        metrics = [
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95"
        ]

        fig, ax = plt.subplots(figsize=(11, 6))

        data.plot(
            x="model",
            y=metrics,
            kind="bar",
            ax=ax
        )

        ax.set_title("Сравнение качества моделей")
        ax.set_xlabel("Метод обнаружения")
        ax.set_ylabel("Значение метрики")
        ax.set_ylim(0, 1)
        ax.legend(title="Метрики")
        ax.grid(axis="y")

        chart = figure_to_image(fig)

        # Дополнительная таблица результатов
        table = data.copy()

        if "inference_time_ms" in table.columns:
            table = table[
                [
                    "model",
                    "precision",
                    "recall",
                    "f1",
                    "map50",
                    "map50_95",
                    "inference_time_ms"
                ]
            ]

        return chart, table, (
            f"Сравнение построено. Количество моделей: {len(data)}."
        )

    except Exception as error:
        return None, None, f"Ошибка обработки результатов: {error}"

def analyze_results(history_file, metrics_file):
    """Обрабатывает оба файла и возвращает результаты для интерфейса."""

    history_chart, history_status = load_training_history(
        history_file
    )

    comparison_chart, comparison_table, comparison_status = (
        load_model_comparison(metrics_file)
    )

    status = (
        "АНАЛИТИКА ОБУЧЕНИЯ\n"
        + history_status
        + "\n\nСРАВНЕНИЕ МОДЕЛЕЙ\n"
        + comparison_status
    )

    return (
        history_chart,
        comparison_chart,
        comparison_table,
        status
    )