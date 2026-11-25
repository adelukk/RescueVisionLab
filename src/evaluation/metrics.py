from typing import Any, Dict, List, Optional, Tuple

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

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_detections(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Однозначно сопоставляет предсказания с истинными рамками.
    Возвращает:
        TP - правильные обнаружения
        FP - ложные обнаружения
        FN - пропущенные объекты
    """
    if len(predictions) != len(confidences):
        raise ValueError("Количество predictions и confidences должно совпадать.")

    if len(predictions) == 0:
        return 0, 0, len(ground_truth)

    order = sorted(range(len(predictions)), key=lambda i: confidences[i], reverse=True)
    matched_gt = [False] * len(ground_truth)

    tp = 0
    fp = 0

    for pred_idx in order:
        pred_box = predictions[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth):
            if matched_gt[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for is_matched in matched_gt if not is_matched)
    return tp, fp, fn


def calculate_precision(tp: int, fp: int) -> float:
    """Точность (Precision) = TP / (TP + FP)."""
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def calculate_recall(tp: int, fn: int) -> float:
    """Полнота (Recall) = TP / (TP + FN)."""
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """F1-мера = 2 * P * R / (P + R)."""
    denom = precision + recall
    return (2.0 * precision * recall) / denom if denom > 0 else 0.0


def calculate_detection_metrics(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Считает TP, FP, FN, Precision, Recall и F1 для заданного порога IoU."""
    tp, fp, fn = match_detections(ground_truth, predictions, confidences, iou_threshold)
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1 = calculate_f1(precision, recall)

    return {
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": iou_threshold,
    }


def compute_pr_curve(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Строит кривую Precision–Recall по предсказаниям, отсортированным
    по убыванию confidence.
    """
    if len(predictions) != len(confidences):
        raise ValueError("Количество predictions и confidences должно совпадать.")

    num_gt = len(ground_truth)
    if num_gt == 0:
        return {
            "precision": [0.0 for _ in predictions],
            "recall": [0.0 for _ in predictions],
            "confidences": sorted(confidences, reverse=True),
        }

    order = sorted(range(len(predictions)), key=lambda i: confidences[i], reverse=True)
    matched_gt = [False] * num_gt

    precisions: List[float] = []
    recalls: List[float] = []
    sorted_confs: List[float] = []

    tp_cum = 0
    fp_cum = 0

    for pred_idx in order:
        pred_box = predictions[pred_idx]
        sorted_confs.append(float(confidences[pred_idx]))

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth):
            if matched_gt[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_gt_idx] = True
            tp_cum += 1
        else:
            fp_cum += 1

        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / num_gt)

    return {
        "precision": precisions,
        "recall": recalls,
        "confidences": sorted_confs,
    }


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """
    Вычисляет Average Precision (AP) по точкам Recall и Precision
    с монотонизацией огибающей precision (в духе COCO/VOC).
    """
    if not recalls or not precisions:
        return 0.0

    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def calculate_map(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Считает AP при порогах 0.50 (AP50), 0.75 (AP75) и среднее mAP50:95
    (по порогам от 0.50 до 0.95 с шагом 0.05).
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    per_threshold: Dict[float, float] = {}
    for thr in iou_thresholds:
        curve = compute_pr_curve(ground_truth, predictions, confidences, iou_threshold=thr)
        per_threshold[thr] = compute_ap(curve["recall"], curve["precision"])

    ap50 = per_threshold.get(0.50)
    if ap50 is None:
        c50 = compute_pr_curve(ground_truth, predictions, confidences, iou_threshold=0.50)
        ap50 = compute_ap(c50["recall"], c50["precision"])

    ap75 = per_threshold.get(0.75)
    if ap75 is None:
        c75 = compute_pr_curve(ground_truth, predictions, confidences, iou_threshold=0.75)
        ap75 = compute_ap(c75["recall"], c75["precision"])

    map_50_95 = sum(per_threshold.values()) / len(per_threshold) if per_threshold else 0.0

    return {
        "AP50": ap50,
        "AP75": ap75,
        "mAP50:95": map_50_95,
        "per_threshold": per_threshold,
    }


def analyze_errors(
    ground_truth: List[Box],
    predictions: List[Box],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Анализирует ошибки и ложные срабатывания (FP / FN):
    - разделяет FP на ошибки неточного позиционирования (low_iou) и фон/дубли (background_fp);
    - возвращает список ложных срабатываний для разбора причин.
    """
    if len(predictions) != len(confidences):
        raise ValueError("Количество predictions и confidences должно совпадать.")

    order = sorted(range(len(predictions)), key=lambda i: confidences[i], reverse=True)
    matched_gt = [False] * len(ground_truth)

    tp = 0
    low_iou_fp = 0
    background_fp = 0
    false_positives: List[Dict[str, Any]] = []

    for pred_idx in order:
        pred_box = predictions[pred_idx]
        conf = float(confidences[pred_idx])

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truth):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        best_free_iou = 0.0
        best_free_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truth):
            if matched_gt[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_free_iou:
                best_free_iou = iou
                best_free_gt_idx = gt_idx

        if best_free_gt_idx >= 0 and best_free_iou >= iou_threshold:
            matched_gt[best_free_gt_idx] = True
            tp += 1
        else:
            if best_iou > 0.0:
                reason = "low_iou_or_duplicate"
                low_iou_fp += 1
            else:
                reason = "background_or_no_overlap"
                background_fp += 1

            false_positives.append(
                {
                    "pred_index": pred_idx,
                    "box": pred_box,
                    "confidence": conf,
                    "max_iou_any_gt": best_iou,
                    "max_iou_free_gt": best_free_iou,
                    "reason": reason,
                }
            )

    missed_gt_indices = [i for i, is_matched in enumerate(matched_gt) if not is_matched]
    fn = len(missed_gt_indices)

    return {
        "confusion_counts": {
            "TP": tp,
            "FP": tp and (low_iou_fp + background_fp) or (low_iou_fp + background_fp),
            "FN": fn,
            "FP_low_iou_or_dup": low_iou_fp,
            "FP_background": background_fp,
        },
        "missed_gt_indices": missed_gt_indices,
        "false_positives": false_positives,
    }