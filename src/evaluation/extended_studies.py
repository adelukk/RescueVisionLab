from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.evaluation.metrics import analyze_errors, calculate_iou, match_detections

MODELS = ["yolo26n", "fasterrcnn", "retinanet"]


def size_category(box: List[float]) -> str:
    x1, y1, x2, y2 = box
    area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
    if area < 32.0 ** 2:
        return "small"
    if area <= 96.0 ** 2:
        return "medium"
    return "large"


def filter_preds(
    boxes: List[List[float]],
    scores: List[float],
    conf_thresh: float = 0.25,
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


def compute_size_recall(
    per_image: List[Dict[str, Any]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict[str, Dict[str, Any]]:
    cats = {
        "small": {"tp": 0, "total": 0},
        "medium": {"tp": 0, "total": 0},
        "large": {"tp": 0, "total": 0},
    }
    for item in per_image:
        gt_boxes = item.get("gt_boxes", []) or []
        pred_boxes, pred_scores = filter_preds(
            item.get("pred_boxes", []) or [],
            item.get("pred_scores", []) or [],
            conf_thresh=conf_thresh,
        )
        gt_cats = [size_category(g) for g in gt_boxes]
        for c in gt_cats:
            cats[c]["total"] += 1

        matched_gt = set()
        order = sorted(
            range(len(pred_boxes)),
            key=lambda i: pred_scores[i],
            reverse=True,
        )
        for p_idx in order:
            best_iou = 0.0
            best_g = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = float(calculate_iou(pred_boxes[p_idx], g_box))
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    best_g = g_idx
            if best_g != -1:
                matched_gt.add(best_g)
                cats[gt_cats[best_g]]["tp"] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for c, stats in cats.items():
        tot = stats["total"]
        tp = stats["tp"]
        result[c] = {
            "count": tot,
            "tp": tp,
            "recall": round(tp / tot, 4) if tot > 0 else None,
        }
    return result


def draw_dashed_rect(
    draw: Any,
    box: List[float],
    color: Tuple[int, int, int],
    width: int = 2,
    dash: int = 8,
    gap: int = 5,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in segments:
        horiz = ay == by
        s0, s1 = (ax, bx) if horiz else (ay, by)
        step = 1 if s1 >= s0 else -1
        pos = s0
        while (pos - s1) * step < 0:
            end = pos + step * dash
            if (end - s1) * step > 0:
                end = s1
            pts = [(pos, ay), (end, by)] if horiz else [(ax, pos), (bx, end)]
            draw.line(pts, fill=color, width=width)
            pos = end + step * gap


def render_example(
    img_path: Path,
    out_path: Path,
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    pred_scores: List[float],
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    if not img_path.exists():
        return False
    with Image.open(img_path) as im:
        canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        # Пунктир — истинная разметка (GT)
        for g in gt_boxes:
            draw_dashed_rect(draw, g, color=(0, 220, 90), width=3)
        # Сплошная рамка — предсказание модели
        for p, s in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = [int(round(v)) for v in p]
            draw.rectangle([x1, y1, x2, y2], outline=(240, 60, 60), width=2)
            draw.text((x1 + 2, max(0, y1 - 12)), f"{s:.2f}", fill=(240, 60, 60))
        canvas.save(out_path)
    return True


def run_extended_studies(max_examples_per_bucket: int = 5) -> None:
    pred_dir = Path("results_final/research/predictions")
    img_dir = Path("datasets/processed/combined_yolo/images/test")
    metrics_out = Path("results_final/research")
    err_root = Path("results_final/research/error_analysis")
    metrics_out.mkdir(parents=True, exist_ok=True)

    size_rows: List[Dict[str, Any]] = []
    size_summary: Dict[str, Any] = {}

    for model_name in MODELS:
        pred_file = pred_dir / f"{model_name}_test.json"
        if not pred_file.exists():
            print(f"[WARN] Missing {pred_file}, skipping {model_name}")
            continue
        data = json.loads(pred_file.read_text(encoding="utf-8"))
        per_image = data.get("per_image", [])

        # Исследование №3: влияние размера объекта
        s_res = compute_size_recall(per_image)
        size_summary[model_name] = s_res
        for cat in ("small", "medium", "large"):
            size_rows.append(
                {
                    "model": model_name,
                    "size_category": cat,
                    "count": s_res[cat]["count"],
                    "tp": s_res[cat]["tp"],
                    "recall": s_res[cat]["recall"],
                }
            )

        # Исследование №2: подборки примеров TP / FP / FN
        buckets: Dict[str, List[Dict[str, Any]]] = {"tp": [], "fp": [], "fn": []}
        for item in per_image:
            img_name = item.get("image", "")
            gt_boxes = item.get("gt_boxes", []) or []
            p_boxes, p_scores = filter_preds(
                item.get("pred_boxes", []) or [],
                item.get("pred_scores", []) or [],
                conf_thresh=0.25,
            )
            tp, fp, fn = match_detections(gt_boxes, p_boxes, p_scores, iou_threshold=0.50)
            err_info = analyze_errors(gt_boxes, p_boxes, p_scores, iou_threshold=0.50)

            base_record = {
                "image": img_name,
                "gt_boxes": gt_boxes,
                "pred_boxes": p_boxes,
                "confidence": p_scores,
                "counts": {"tp": tp, "fp": fp, "fn": fn},
                "error_analysis": err_info,
            }
            if tp > 0 and len(buckets["tp"]) < max_examples_per_bucket:
                buckets["tp"].append({**base_record, "bucket": "tp"})
            if fp > 0 and len(buckets["fp"]) < max_examples_per_bucket:
                buckets["fp"].append({**base_record, "bucket": "fp"})
            if fn > 0 and len(buckets["fn"]) < max_examples_per_bucket:
                buckets["fn"].append({**base_record, "bucket": "fn"})

        for b_name in ("tp", "fp", "fn"):
            b_dir = err_root / model_name / b_name
            b_dir.mkdir(parents=True, exist_ok=True)
            for idx, rec in enumerate(buckets[b_name], start=1):
                stem = Path(rec["image"]).stem
                out_img = b_dir / f"{idx:02d}_{stem}.png"
                rendered = render_example(
                    img_dir / rec["image"],
                    out_img,
                    rec["gt_boxes"],
                    rec["pred_boxes"],
                    rec["confidence"],
                )
                rec["rendered_image"] = str(out_img) if rendered else None
            (b_dir / "examples.json").write_text(
                json.dumps(buckets[b_name], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(
            f"[{model_name}] size recall: {s_res} | "
            f"error buckets: tp={len(buckets['tp'])}, fp={len(buckets['fp'])}, fn={len(buckets['fn'])}"
        )

    (metrics_out / "object_size_recall.json").write_text(
        json.dumps(size_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (metrics_out / "object_size_recall.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "size_category", "count", "tp", "recall"])
        writer.writeheader()
        writer.writerows(size_rows)

    print("Done. Saved error analysis to results/error_analysis and object size recall to results/metrics/object_size_recall.csv")


if __name__ == "__main__":
    run_extended_studies()