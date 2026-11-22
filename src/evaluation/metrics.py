from typing import List, Tuple, Dict


Box = List[float]


def calculate_iou(box1: Box, box2: Box) -> float:
    """
    Вычисляет Intersection over Union (IoU)
    для двух рамок формата [x1, y1, x2, y2].
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )

    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_detections(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Однозначно сопоставляет предсказания с истинными рамками.

    Возвращает:
        TP - правильные обнаружения
        FP - ложные обнаружения
        FN - пропущенные объекты
    """

    if len(predictions) != len(confidences):
        raise ValueError(
            "Количество predictions и confidences должно совпадать."
        )

    # Обрабатываем предсказания от самых уверенных
    # к менее уверенным.
    order = sorted(
        range(len(predictions)),
        key=lambda i: confidences[i],
        reverse=True
    )

    matched_gt = set()

    tp = 0
    fp = 0

    for prediction_index in order:

        prediction = predictions[prediction_index]

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt_box in enumerate(ground_truth):

            if gt_index in matched_gt:
                continue

            iou = calculate_iou(prediction, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if (
            best_gt_index is not None
            and best_iou >= iou_threshold
        ):
            tp += 1
            matched_gt.add(best_gt_index)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    return tp, fp, fn


def calculate_precision(tp: int, fp: int) -> float:
    """
    Precision = TP / (TP + FP)
    """

    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def calculate_recall(tp: int, fn: int) -> float:
    """
    Recall = TP / (TP + FN)
    """

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def calculate_f1(precision: float, recall: float) -> float:
    """
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """

    denominator = precision + recall

    if denominator == 0:
        return 0.0

    return (
        2.0 * precision * recall
        / denominator
    )


def calculate_detection_metrics(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5
) -> Dict[str, float]:

    tp, fp, fn = match_detections(
        ground_truth,
        predictions,
        confidences,
        iou_threshold
    )

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": iou_threshold,
    }