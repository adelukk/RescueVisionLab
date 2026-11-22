from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    epochs,
    train_loss,
    val_loss,
    output_path
):
    """
    График изменения train и validation loss.
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Train loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation loss"
    )

    plt.xlabel("Эпоха")
    plt.ylabel("Loss")

    plt.title(
        "Изменение функции потерь в процессе обучения"
    )

    plt.grid(True)
    plt.legend()

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


def plot_metrics(
    names,
    values,
    title,
    output_path
):
    """
    Столбчатая диаграмма метрик.
    """

    plt.figure(figsize=(10, 6))

    plt.bar(
        names,
        values
    )

    plt.ylabel("Значение")

    plt.ylim(
        0,
        1
    )

    plt.title(title)

    plt.grid(
        axis="y"
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    print(
        "Модуль построения графиков загружен."
    )