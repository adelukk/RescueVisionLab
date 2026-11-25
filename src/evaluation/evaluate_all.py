import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from src.evaluation.metrics import (
    calculate_detection_metrics,
    calculate_f1,
    calculate_map,
    calculate_precision,
    calculate_recall,
    match_detections,
)
from src.inference.predictor import MODEL_NAMES, clear_model_cache, predict

DEFAULT_DATASET_ROOT = Path("datasets/processed/combined_yolo")
DEFAULT_RESULTS_DIR = Path("results_final/research")
IOU_THRESHOLDS_MAP = [round(0.50 + 0.05 * i, 2) for i in range(10)]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_yolo_ground_truth(
    label_path: Path, img_width: int, img_height: int
) -> List[List[float]]:
    gt_boxes: List[List[float]] = []
    if not label_path.is_file():
        return gt_boxes
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return gt_boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id != 0:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        x1 = max(0.0, (xc - w / 2.0) * img_width)
        y1 = max(0.0, (yc - h / 2.0) * img_height)
        x2 = min(float(img_width), (xc + w / 2.0) * img_width)
        y2 = min(float(img_height), (yc + h / 2.0) * img_height)
        if x2 > x1 and y2 > y1:
            gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes


def resolve_split_dir(base_dir: Path, split: str) -> Path:
    if split == "validation":
        split = "val"
    candidates = [split]
    if split == "validation":
        candidates.append("val")
    elif split == "val":
        candidates.append("validation")
    for name in candidates:
        p = base_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Directory for split '{split}' not found in {base_dir}")


def get_split_images(dataset_root: Path, split: str) -> List[Path]:
    img_dir = resolve_split_dir(dataset_root / "images", split)
    return [
        p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def filter_detections(
    boxes: List[List[float]],
    scores: List[float],
    labels: List[int],
    conf_thr: float,
    max_det: int = 100,
) -> Tuple[List[List[float]], List[float]]:
    paired: List[Tuple[float, List[float]]] = []
    for b, s, l in zip(boxes, scores, labels):
        if int(l) == 0 and float(s) >= conf_thr:
            paired.append((float(s), [float(v) for v in b]))
    paired.sort(key=lambda item: item[0], reverse=True)
    paired = paired[:max_det]
    return [item[1] for item in paired], [item[0] for item in paired]


def run_model_on_split(
    model_name: str,
    split: str,
    dataset_root: Path,
    checkpoint_root: Optional[Path],
    low_conf: float = 0.001,
    max_det: int = 100,
) -> Dict[str, Any]:
    img_paths = get_split_images(dataset_root, split)
    label_dir = resolve_split_dir(dataset_root / "labels", split)

    per_image: List[Dict[str, Any]] = []
    all_gt: List[List[float]] = []
    all_pred_boxes: List[List[float]] = []
    all_pred_scores: List[float] = []
    pred_boxes_025: List[List[float]] = []
    pred_scores_025: List[float] = []
    total_ms = 0.0

    for idx, img_path in enumerate(img_paths, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        gt_boxes = load_yolo_ground_truth(label_dir / f"{img_path.stem}.txt", w, h)
        res = predict(
            img_path,
            model_name=model_name,
            confidence=low_conf,
            checkpoint_root=checkpoint_root,
        )
        boxes, scores = filter_detections(
            res["boxes"],
            res["scores"],
            res["labels"],
            conf_thr=low_conf,
            max_det=max_det,
        )
        ms = float(res.get("inference_ms", 0.0))
        total_ms += ms

        all_gt.extend(gt_boxes)
        all_pred_boxes.extend(boxes)
        all_pred_scores.extend(scores)

        for b, s in zip(boxes, scores):
            if s >= 0.25:
                pred_boxes_025.append(b)
                pred_scores_025.append(s)

        per_image.append(
            {
                "image": img_path.name,
                "width": w,
                "height": h,
                "inference_ms": ms,
                "gt_boxes": gt_boxes,
                "pred_boxes": boxes,
                "pred_scores": scores,
            }
        )
        if idx % 33 == 0 or idx == len(img_paths):
            print(f"  [{model_name} | {split}] {idx}/{len(img_paths)} images processed")

    # Финальный mAP уже рассчитан отдельными финальными evaluators.
    # Здесь повторно его не считаем: для research нужны per-image predictions.
    det_025 = calculate_detection_metrics(
        all_gt, pred_boxes_025, pred_scores_025, iou_threshold=0.50
    )
    mean_ms = total_ms / len(img_paths) if img_paths else 0.0

    return {
        "model": model_name,
        "split": split,
        "num_images": len(img_paths),
        "num_gt": len(all_gt),
        "mean_inference_ms": round(mean_ms, 2),
        "TP_0.25": int(det_025["TP"]),
        "FP_0.25": int(det_025["FP"]),
        "FN_0.25": int(det_025["FN"]),
        "precision_0.25": round(det_025["precision"], 4),
        "recall_0.25": round(det_025["recall"], 4),
        "f1_0.25": round(det_025["f1"], 4),
        "AP50": None,
        "AP75": None,
        "mAP50_95": None,
        "per_image": per_image,
    }


def find_best_conf_on_val(
    val_record: Dict[str, Any], iou_thr: float = 0.50
) -> Tuple[float, Dict[str, float]]:
    thresholds = [round(0.05 * i, 2) for i in range(1, 20)]
    best_thr = 0.25
    best_f1 = -1.0
    best_stats: Dict[str, float] = {
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    for thr in thresholds:
        tp_sum = fp_sum = fn_sum = 0
        for img in val_record["per_image"]:
            b_f = [b for b, s in zip(img["pred_boxes"], img["pred_scores"]) if s >= thr]
            s_f = [s for s in img["pred_scores"] if s >= thr]
            tp, fp, fn = match_detections(img["gt_boxes"], b_f, s_f, iou_threshold=iou_thr)
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        p = calculate_precision(tp_sum, fp_sum)
        r = calculate_recall(tp_sum, fn_sum)
        f1 = calculate_f1(p, r)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_stats = {
                "TP": tp_sum,
                "FP": fp_sum,
                "FN": fn_sum,
                "precision": p,
                "recall": r,
                "f1": f1,
            }

    return best_thr, best_stats


def eval_at_conf(
    record: Dict[str, Any], conf_thr: float, iou_thr: float = 0.50
) -> Dict[str, Any]:
    tp_sum = fp_sum = fn_sum = 0
    for img in record["per_image"]:
        b_f = [b for b, s in zip(img["pred_boxes"], img["pred_scores"]) if s >= conf_thr]
        s_f = [s for s in img["pred_scores"] if s >= conf_thr]
        tp, fp, fn = match_detections(img["gt_boxes"], b_f, s_f, iou_threshold=iou_thr)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
    p = calculate_precision(tp_sum, fp_sum)
    r = calculate_recall(tp_sum, fn_sum)
    f1 = calculate_f1(p, r)
    return {
        "conf_thr": conf_thr,
        "TP": tp_sum,
        "FP": fp_sum,
        "FN": fn_sum,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def main(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    checkpoint_root: Optional[Path] = None,
) -> None:
    preds_dir = results_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        print(f"=== Evaluating {model_name} ===")
        val_rec = run_model_on_split(model_name, "validation", dataset_root, checkpoint_root)
        test_rec = run_model_on_split(model_name, "test", dataset_root, checkpoint_root)

        (preds_dir / f"{model_name}_validation.json").write_text(
            json.dumps(val_rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (preds_dir / f"{model_name}_test.json").write_text(
            json.dumps(test_rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        best_conf, _ = find_best_conf_on_val(val_rec, iou_thr=0.50)
        val_tuned = eval_at_conf(val_rec, best_conf, iou_thr=0.50)
        test_tuned = eval_at_conf(test_rec, best_conf, iou_thr=0.50)

        val_row = {k: v for k, v in val_rec.items() if k != "per_image"}
        val_row["best_conf_val"] = best_conf
        val_row["tuned_f1_val"] = val_tuned["f1"]
        val_rows.append(val_row)

        test_row = {k: v for k, v in test_rec.items() if k != "per_image"}
        test_row["tuned_conf"] = best_conf
        test_row["tuned_precision"] = test_tuned["precision"]
        test_row["tuned_recall"] = test_tuned["recall"]
        test_row["tuned_f1"] = test_tuned["f1"]
        test_rows.append(test_row)

        summary_rows.append(
            {
                "model": model_name,
                "precision": test_rec["precision_0.25"],
                "recall": test_rec["recall_0.25"],
                "f1": test_rec["f1_0.25"],
                "AP50": test_rec["AP50"],
                "AP75": test_rec["AP75"],
                "mAP50_95": test_rec["mAP50_95"],
                "tuned_conf": best_conf,
                "tuned_precision": test_tuned["precision"],
                "tuned_recall": test_tuned["recall"],
                "tuned_f1": test_tuned["f1"],
                "mean_inference_ms": test_rec["mean_inference_ms"],
            }
        )

        clear_model_cache()
        gc.collect()

    pd.DataFrame(val_rows).to_csv(results_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(test_rows).to_csv(results_dir / "test_metrics.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "models_comparison.csv", index=False)
    (results_dir / "models_comparison.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done. Saved metrics to", results_dir)


if __name__ == "__main__":
    main()