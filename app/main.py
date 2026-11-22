from app.services.analytics import analyze_results
import gradio as gr

from app.services.detector import (
    AVAILABLE_MODELS,
    detect_people,
)
from app.services.preprocessing import preprocess_image

def process_image(image, model_name, confidence, nms_iou, image_size):

    if image is None:
        return (
            None,
            None,
            [],
            "Пожалуйста, загрузите изображение."
        )

    try:
        # Этап 1. Предобработка
        processed_image, preprocessing_info = preprocess_image(
            image,
            image_size
        )

        # Этап 2. Обнаружение
        result = detect_people(
            processed_image,
            model_name,
            confidence,
            nms_iou
        )

        # Этап 3. Результаты
        detections_table = [
            [
                index + 1,
                detection.get("class", "person"),
                detection.get("confidence", 0),
                detection.get("x1", 0),
                detection.get("y1", 0),
                detection.get("x2", 0),
                detection.get("y2", 0),
            ]
            for index, detection in enumerate(result["detections"])
        ]

        status = (
            "ПРЕДОБРАБОТКА ВЫПОЛНЕНА\n"
            + preprocessing_info
            + "\n\n"
            + result["message"]
        )

        return (
            processed_image,
            result["image"],
            detections_table,
            status
        )

    except Exception as error:
        return None, None, [], f"Ошибка: {error}"

with gr.Blocks(title="RescueVisionLab") as demo:

    gr.Markdown(
        """
        # RescueVisionLab
        ### Система обнаружения людей в спасательных операциях

        Загрузка изображения, настройка параметров и анализ результата.
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                label="Исходное изображение",
                type="pil"
            )

            model_choice = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value="YOLO",
                label="Метод обнаружения"
            )

            confidence_slider = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Порог уверенности (Confidence)"
            )

            iou_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.05,
                label="Порог NMS IoU"
            )

            image_size_slider = gr.Slider(
                minimum=320,
                maximum=1280,
                value=640,
                step=32,
                label="Размер изображения"
            )

            run_button = gr.Button(
                "Запустить обработку",
                variant="primary"
            )

        with gr.Column():

            processed_image = gr.Image(
                label="Изображение после предобработки",
                type="pil"
            )

            result_image = gr.Image(
                label="Результат обработки",
                type="pil"
            )

    gr.Markdown("## Результаты обнаружения")

    detections_table = gr.Dataframe(
        headers=[
            "№",
            "Класс",
            "Confidence",
            "X1",
            "Y1",
            "X2",
            "Y2"
        ],
        label="Обнаруженные объекты",
        interactive=False
    )

    status_output = gr.Textbox(
        label="Этапы обработки и состояние модели",
        lines=8
    )

    run_button.click(
        fn=process_image,
        inputs=[
            input_image,
            model_choice,
            confidence_slider,
            iou_slider,
            image_size_slider
        ],
        outputs=[
            processed_image,
            result_image,
            detections_table,
            status_output
        ]
    )

    gr.Markdown("---")
    gr.Markdown(
        """
        # Аналитика и оценка моделей

        Загрузка результатов экспериментов,
        визуализация обучения и сравнение методов обнаружения.
        """
    )

    with gr.Tab("История обучения"):

        gr.Markdown(
            """
            Загрузите CSV с результатами обучения.
            Необходимые столбцы: epoch, train_loss, val_loss.
            """
        )

        history_file = gr.File(
            label="CSV: история обучения",
            file_types=[".csv"],
            type="filepath"
        )

        history_chart = gr.Image(
            label="График Train Loss и Validation Loss"
        )

    with gr.Tab("Сравнение моделей"):

        gr.Markdown(
            """
            Загрузите CSV с результатами оценки пяти методов.
            Значения метрик должны находиться в диапазоне от 0 до 1.
            """
        )

        metrics_file = gr.File(
            label="CSV: результаты оценки моделей",
            file_types=[".csv"],
            type="filepath"
        )

        comparison_chart = gr.Image(
            label="Сравнение качества моделей"
        )

        comparison_table = gr.Dataframe(
            label="Таблица сравнительной оценки",
            interactive=False
        )

    analytics_status = gr.Textbox(
        label="Состояние аналитики",
        lines=5
    )

    analytics_button = gr.Button(
        "Построить графики и таблицу",
        variant="primary"
    )

    analytics_button.click(
        fn=analyze_results,
        inputs=[
            history_file,
            metrics_file
        ],
        outputs=[
            history_chart,
            comparison_chart,
            comparison_table,
            analytics_status
        ]
    )
if __name__ == "__main__":
    demo.launch()