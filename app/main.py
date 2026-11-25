import os
import tempfile
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from app.services.analytics import analyze_results
from app.services.detector import AVAILABLE_MODELS, detect_people
from app.services.preprocessing import preprocess_image

# --------------------------------------------------
# ОБРАБОТКА ФОТОГРАФИЙ
# --------------------------------------------------
def save_original_image(image):
    return image.copy() if image is not None else None


def process_image(image, model_name, confidence, nms_iou, image_size):
    if image is None:
        return None, [], "Пожалуйста, загрузите изображение."

    try:
        processed_image, preprocessing_info = preprocess_image(
            image.copy(),
            image_size
        )

        result = detect_people(
            processed_image,
            model_name,
            confidence,
            nms_iou
        )

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
            "Предобработка выполнена\n"
            + preprocessing_info
            + "\n\n"
            + result["message"]
        )

        return result["image"], detections_table, status

    except Exception as error:
        return None, [], f"Ошибка: {error}"

# --------------------------------------------------
# ОБРАБОТКА ВИДЕО ПО КАДРАМ
# --------------------------------------------------
def process_video(
    video_path,
    model_name,
    confidence,
    nms_iou,
    image_size,
    frame_step
):
    if video_path is None:
        return None, [], "Пожалуйста, загрузите видео."

    cap = None
    writer = None
    output_path = None

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, [], "Не удалось открыть видео."

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if original_width <= 0 or original_height <= 0:
            return None, [], "Не удалось определить размер видео."

        # Создаём выходной видеофайл
        file_descriptor, output_path = tempfile.mkstemp(
            suffix=".mp4"
        )
        os.close(file_descriptor)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (original_width, original_height)
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Не удалось создать выходное видео."
            )

        frame_index = 0
        processed_frames = 0
        total_detections = 0
        detections_table = []

        # Последовательно читаем кадры видео
        while True:

            success, frame = cap.read()

            if not success:
                break

            # Анализируем каждый N-й кадр
            if frame_index % int(frame_step) == 0:

                # OpenCV BGR -> RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                pil_image = Image.fromarray(rgb_frame)

                # Предобработка кадра
                processed_image, _ = preprocess_image(
                    pil_image,
                    image_size
                )

                # Обнаружение людей
                result = detect_people(
                    processed_image,
                    model_name,
                    confidence,
                    nms_iou
                )

                detections = result["detections"]

                processed_frames += 1
                total_detections += len(detections)

                # Записываем найденные объекты в таблицу
                timestamp = frame_index / fps

                for detection in detections:

                    detections_table.append(
                        [
                            frame_index,
                            round(timestamp, 2),
                            detection.get("class", "person"),
                            round(
                                float(
                                    detection.get("confidence", 0)
                                ),
                                4
                            ),
                            detection.get("x1", 0),
                            detection.get("y1", 0),
                            detection.get("x2", 0),
                            detection.get("y2", 0),
                        ]
                    )

                # Получаем кадр с рамками
                annotated = result["image"]

                if isinstance(annotated, Image.Image):
                    annotated = np.array(
                        annotated.convert("RGB")
                    )
                else:
                    annotated = np.asarray(annotated)

                # RGB -> BGR
                annotated = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_RGB2BGR
                )

                # Возвращаем исходный размер видео
                if (
                    annotated.shape[1] != original_width
                    or annotated.shape[0] != original_height
                ):
                    annotated = cv2.resize(
                        annotated,
                        (original_width, original_height)
                    )

                writer.write(annotated)

            else:
                # Пропущенные кадры сохраняем без изменений
                writer.write(frame)

            frame_index += 1

        cap.release()
        cap = None

        writer.release()
        writer = None

        if frame_index == 0:
            raise RuntimeError(
                "В видео не найдено ни одного кадра."
            )

        status = (
            "Обработка видео завершена!\n\n"
            f"Всего кадров в видео: {frame_index}\n"
            f"Проанализировано кадров: {processed_frames}\n"
            f"Найдено людей: {total_detections}\n"
            f"Частота обработки: каждый {frame_step}-й кадр\n"
            f"Продолжительность: {frame_index / fps:.2f} сек."
        )

        return output_path, detections_table, status

    except Exception as error:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        if output_path and os.path.exists(output_path):
            os.remove(output_path)

        return None, [], f"Ошибка обработки видео: {error}"

# --------------------------------------------------
# ИНТЕРФЕЙС RESCUEVISIONLAB
# --------------------------------------------------
with gr.Blocks(title="RescueVisionLab") as demo:
    gr.Markdown(
        """
        # RescueVisionLab

        ### Система обнаружения людей в спасательных операциях

        Загрузка изображений и видео, обнаружение людей
        и анализ результатов работы моделей.
        """
    )

    # --------------------------------------------------
    # ВКЛАДКА ФОТОГРАФИЙ
    # --------------------------------------------------

    with gr.Tab("Обработка фотографии"):

        gr.Markdown(
            """
            Загрузите фотографию, выберите модель и параметры.
            После обработки на изображении появятся рамки.
            """
        )

        input_image = gr.Image(
            label="Фотография",
            type="pil"
        )

        model_choice = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="YOLO",
            label="Модель обнаружения"
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
            "Обнаружить людей",
            variant="primary"
        )

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
            label="Статус обработки",
            lines=5
        )

        original_image = gr.State()
        input_image.upload(
            fn=save_original_image,
            inputs=input_image,
            outputs=original_image,
            queue=False
        )
        input_image.clear(
            fn=lambda: None,
            outputs=original_image,
            queue=False
        )

        run_button.click(
            fn=process_image,
            inputs=[
                original_image,
                model_choice,
                confidence_slider,
                iou_slider,
                image_size_slider
            ],
            outputs=[
                input_image,
                detections_table,
                status_output
            ]
        )

    # --------------------------------------------------
    # ВКЛАДКА ВИДЕО
    # --------------------------------------------------

    with gr.Tab("Анализ видео"):

        gr.Markdown(
            """
            ## Покадровый анализ видео

            Загрузите видео с места происшествия.
            Система последовательно обработает выбранные кадры
            и отметит обнаруженных людей рамками.
            """
        )

        video_input = gr.Video(
            label="Загрузить видео"
        )

        video_model_choice = gr.Dropdown(
            choices=AVAILABLE_MODELS,
            value="YOLO",
            label="Модель обнаружения"
        )

        video_confidence = gr.Slider(
            minimum=0.05,
            maximum=0.95,
            value=0.25,
            step=0.05,
            label="Порог уверенности (Confidence)"
        )

        video_iou = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            value=0.5,
            step=0.05,
            label="Порог NMS IoU"
        )

        video_image_size = gr.Slider(
            minimum=320,
            maximum=1280,
            value=640,
            step=32,
            label="Размер кадра"
        )

        frame_step_slider = gr.Slider(
            minimum=1,
            maximum=30,
            value=1,
            step=1,
            label="Обрабатывать каждый N-й кадр"
        )

        video_run_button = gr.Button(
            "Начать анализ видео",
            variant="primary"
        )

        video_status = gr.Textbox(
            label="Статус анализа видео",
            lines=7
        )

        gr.Markdown("### Результат обработки")

        video_output = gr.Video(
            label="Видео с рамками обнаружения"
        )

        video_detections_table = gr.Dataframe(
            headers=[
                "Кадр",
                "Время (сек.)",
                "Класс",
                "Confidence",
                "X1",
                "Y1",
                "X2",
                "Y2"
            ],
            label="Обнаруженные люди по кадрам",
            interactive=False
        )

        video_run_button.click(
            fn=process_video,
            inputs=[
                video_input,
                video_model_choice,
                video_confidence,
                video_iou,
                video_image_size,
                frame_step_slider
            ],
            outputs=[
                video_output,
                video_detections_table,
                video_status
            ]
        )

    # --------------------------------------------------
    # ВКЛАДКА АНАЛИТИКИ
    # --------------------------------------------------

    with gr.Tab("Аналитика"):

        gr.Markdown(
            """
            ## Аналитика и оценка моделей

            Загрузка результатов экспериментов,
            визуализация обучения и сравнение качества моделей.
            """
        )

        with gr.Tab("История обучения"):

            gr.Markdown(
                """
                Загрузите CSV с историей обучения.

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
                Загрузите CSV с результатами оценки моделей.

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
