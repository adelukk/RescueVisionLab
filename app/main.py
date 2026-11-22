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

if __name__ == "__main__":
    demo.launch()