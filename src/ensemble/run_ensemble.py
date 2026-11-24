from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.ensemble.fusion import weighted_boxes_fusion
from src.evaluation.metrics import calculate_detection_metrics, calculate_map


def load_predictions(pred_path: Path) -> Dict[str, Any]:
    return json.loads(pred_path.read_text(encoding="utf-8"))


def filter_top_k(
    boxes: List[List[float]],
    scores: List[float],
    conf_thresh: float = 0.001,
    max_det: int = 100,
) -> Tuple[List[List[float]], List[float]]:
    pairs = [
        (b, float(s))
        for b, s in zip(boxes, scores)
        if float(s) >= conf_thresh
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    pairs = pairs[:max_det]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def fuse_split(
    model_data_list: List[Dict[str, Any]],
    weights: Sequence[float],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    max_det: int = 100,
) -> Tuple[List[Dict[str, Any]], float]:
    per_image_maps = [d.get("per_image", []) for d in model_data_list]
    num_images = len(per_image_maps[0])
    fused_per_image: List[Dict[str, Any]] = []
    total_fusion_ms = 0.0

    for img_idx in range(num_images):
        base_item = per_image_maps[0][img_idx]
        gt_boxes = base_item.get("gt_boxes", []) or []

        boxes_per_model: List[List[List[float]]] = []
        scores_per_model: List[List[float]] = []
        prev_infer_ms = 0.0

        for m_idx, m_per_img in enumerate(per_image_maps):
            it = m_per_img[img_idx]
            pb, ps = filter_top_k(
                it.get("pred_boxes", []) or [],
                it.get("pred_scores", []) or [],
                conf_thresh=skip_box_thr,
                max_det=max_det,
            )
            boxes_per_model.append(pb)
            scores_per_model.append(ps)
            prev_infer_ms += float(it.get("inference_ms", 0.0))

        t0 = time.perf_counter()
        f_boxes, f_scores = weighted_boxes_fusion(
            boxes_list=boxes_per_model,
            scores_list=scores_per_model,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        f_boxes, f_scores = filter_top_k(f_boxes, f_scores, conf_thresh=skip_box_thr, max_det=max_det)
        fusion_ms = (time.perf_counter() - t0) * 1000.0
        total_fusion_ms += fusion_ms

        fused_per_image.append(
            {
                "image": base_item.get("image", ""),
                "gt_boxes": gt_boxes,
                "pred_boxes": f_boxes,
                "pred_scores": f_scores,
                "fusion_ms": round(fusion_ms, 4),
                "combined_model_infer_ms": round(prev_infer_ms, 2),
            }
        )

    mean_fusion_ms = total_fusion_ms / max(1, num_images)
    return fused_per_image, mean_fusion_ms


def evaluate_fused(
    fused_per_image: List[Dict[str, Any]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    ap50_vals: List[float] = []
    ap75_vals: List[float] = []
    map50_95_vals: List[float] = []

    for it in fused_per_image:
        gt = it.get("gt_boxes", []) or []
        pb_all = it.get("pred_boxes", []) or []
        ps_all = [float(s) for s in (it.get("pred_scores", []) or [])]

        if gt or pb_all:
            m_res = calculate_map(gt, pb_all, ps_all)
            ap50_vals.append(float(m_res.get("AP50", m_res.get("ap50", 0.0))))
            ap75_vals.append(float(m_res.get("AP75", m_res.get("ap75", 0.0))))
            map50_95_vals.append(
                float(m_res.get("mAP50_95", m_res.get("mAP50:95", m_res.get("map_50_95", 0.0))))
            )

        pb_thr, ps_thr = filter_top_k(pb_all, ps_all, conf_thresh=conf_thresh, max_det=100)
        det = calculate_detection_metrics(gt, pb_thr, ps_thr, iou_threshold=iou_thresh)
        total_tp += int(det.get("TP", det.get("tp", 0)))
        total_fp += int(det.get("FP", det.get("fp", 0)))
        total_fn += int(det.get("FN", det.get("fn", 0)))

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    avg = lambda arr: (sum(arr) / len(arr)) if arr else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "AP50": round(avg(ap50_vals), 4),
        "AP75": round(avg(ap75_vals), 4),
        "mAP50_95": round(avg(map50_95_vals), 4),
    }

    det_res = calculate_detection_metrics(gt_list, pred_thr, score_thr, iou_threshold=iou_thresh)

    ap50 = float(map_res.get("AP50", map_res.get("ap50", 0.0)))
    ap75 = float(map_res.get("AP75", map_res.get("ap75", 0.0)))
    map_50_95 = float(
        map_res.get("mAP50_95", map_res.get("mAP50:95", map_res.get("map_50_95", 0.0)))
    )

    return {
        "precision": round(float(det_res.get("precision", 0.0)), 4),
        "recall": round(float(det_res.get("recall", 0.0)), 4),
        "f1": round(float(det_res.get("f1", 0.0)), 4),
        "AP50": round(ap50, 4),
        "AP75": round(ap75, 4),
        "mAP50_95": round(map_50_95, 4),
    }


def main() -> None:
    pred_dir = Path("results/metrics/predictions")
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    configs: List[Tuple[str, List[str], List[float], float]] = [
        ("two_models_yolo26n_retinanet", ["yolo26n", "retinanet"], [0.6, 0.4], 0.55),
        ("two_models_yolo26n_fcos", ["yolo26n", "fcos"], [0.65, 0.35], 0.55),
        ("three_models_yolo26n_retinanet_fcos", ["yolo26n", "retinanet", "fcos"], [0.5, 0.3, 0.2], 0.55),
        ("all_five_models_weighted", ["yolo26n", "retinanet", "fasterrcnn", "fcos", "ssd300"], [0.45, 0.25, 0.15, 0.1, 0.05], 0.55),
    ]

    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for ens_name, models, weights, iou_thr in configs:
        val_files = [pred_dir / f"{m}_validation.json" for m in models]
        test_files = [pred_dir / f"{m}_test.json" for m in models]
        if not all(p.exists() for p in val_files + test_files):
            print(f"[WARN] Missing prediction files for {ens_name}, skipping.")
            continue

        val_data = [load_predictions(p) for p in val_files]
        test_data = [load_predictions(p) for p in test_files]

        val_fused, _ = fuse_split(val_data, weights=weights, iou_thr=iou_thr)

        # Подбираем лучший conf на validation по максимальному F1
        best_conf = 0.25
        best_val_f1 = -1.0
        for cand_conf in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            m_val = evaluate_fused(val_fused, conf_thresh=cand_conf)
            if m_val["f1"] > best_val_f1:
                best_val_f1 = m_val["f1"]
                best_conf = cand_conf

        test_fused, mean_fusion_ms = fuse_split(test_data, weights=weights, iou_thr=iou_thr)
        test_metrics_025 = evaluate_fused(test_fused, conf_thresh=0.25)
        test_metrics_tuned = evaluate_fused(test_fused, conf_thresh=best_conf)

        row = {
            "ensemble": ens_name,
            "models": "+".join(models),
            "weights": ",".join(str(x) for x in weights),
            "iou_thr": iou_thr,
            "best_conf_val": best_conf,
            "val_f1_at_best_conf": round(best_val_f1, 4),
            "test_precision_0.25": test_metrics_025["precision"],
            "test_recall_0.25": test_metrics_025["recall"],
            "test_f1_0.25": test_metrics_025["f1"],
            "test_AP50": test_metrics_025["AP50"],
            "test_AP75": test_metrics_025["AP75"],
            "test_mAP50_95": test_metrics_025["mAP50_95"],
            "test_tuned_f1": test_metrics_tuned["f1"],
            "mean_fusion_ms": round(mean_fusion_ms, 3),
        }
        rows.append(row)
        summary[ens_name] = {**row, "test_at_tuned_conf": test_metrics_tuned}
        print(
            f"[{ens_name}] best_conf_val={best_conf} | "
            f"test P={test_metrics_025['precision']} R={test_metrics_025['recall']} F1={test_metrics_025['f1']} | "
            f"AP50={test_metrics_025['AP50']} mAP50:95={test_metrics_025['mAP50_95']} | "
            f"fusion_ms={mean_fusion_ms:.3f}"
        )

    (out_dir / "ensemble_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (out_dir / "ensemble_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved ensemble results to results/metrics/ensemble_comparison.csv")


if __name__ == "__main__":
    main()