from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.metrics import calculate_iou


def weighted_boxes_fusion(
    boxes_list: List[List[List[float]]],
    scores_list: List[List[float]],
    weights: Optional[Sequence[float]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    sigma: float = 0.0,
    conf_type: str = "avg",
) -> Tuple[List[List[float]], List[float]]:
    """
    Weighted Boxes Fusion (WBF) для списков предсказаний нескольких моделей.
    Координаты рамок передаются в пикселях [x1, y1, x2, y2].
    """
    num_models = len(boxes_list)
    if num_models == 0:
        return [], []

    if weights is None:
        w = np.ones(num_models, dtype=np.float64)
    else:
        if len(weights) != num_models:
            raise ValueError("Длина weights должна совпадать с количеством моделей.")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w <= 0):
            raise ValueError("Все веса моделей должны быть положительными.")

    w = w / np.sum(w)

    # Собираем все рамки с учётом веса модели и порога отсечения
    candidates: List[Dict[str, Any]] = []
    for m_idx in range(num_models):
        m_boxes = boxes_list[m_idx] or []
        m_scores = scores_list[m_idx] or []
        m_weight = float(w[m_idx])
        for box, score in zip(m_boxes, m_scores):
            sc = float(score)
            if sc < skip_box_thr:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": sc,
                    "weighted_score": sc * m_weight,
                    "model_idx": m_idx,
                    "model_weight": m_weight,
                }
            )

    if not candidates:
        return [], []

    # Сортируем по взвешенной уверенности по убыванию
    candidates.sort(key=lambda c: c["weighted_score"], reverse=True)

    # Кластеры: каждый кластер хранит представителя (усреднённая рамка) и список входящих рамок
    clusters: List[Dict[str, Any]] = []
    for cand in candidates:
        best_cluster_idx = -1
        best_iou = 0.0
        for c_idx, cl in enumerate(clusters):
            iou = calculate_iou(cand["box"], cl["box"])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_cluster_idx = c_idx

        if best_cluster_idx >= 0:
            cl = clusters[best_cluster_idx]
            cl["members"].append(cand)
            total_w = sum(m["weighted_score"] for m in cl["members"])
            if total_w > 0:
                acc = np.zeros(4, dtype=np.float64)
                for m in cl["members"]:
                    acc += np.asarray(m["box"], dtype=np.float64) * m["weighted_score"]
                cl["box"] = (acc / total_w).tolist()
        else:
            clusters.append(
                {
                    "box": list(cand["box"]),
                    "members": [cand],
                }
            )

    fused_boxes: List[List[float]] = []
    fused_scores: List[float] = []

    for cl in clusters:
        members = cl["members"]
        if conf_type == "max":
            final_score = max(m["score"] for m in members)
        else:
            # Нормируем сумму скоров на суммарный вес всех моделей в ансамбле (или вошедших моделей)
            score_sum = sum(m["weighted_score"] for m in members)
            final_score = float(np.clip(score_sum, 0.0, 1.0))

        fused_boxes.append([round(float(v), 4) for v in cl["box"]])
        fused_scores.append(round(final_score, 6))

    # Сортируем итоговые рамки по убыванию уверенности
    order = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)
    return [fused_boxes[i] for i in order], [fused_scores[i] for i in order]