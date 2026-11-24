from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.evaluation.evaluate_all import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_RESULTS_DIR,
    filter_detections,
    get_split_images,
    load_yolo_ground_truth,
    resolve_split_dir,
)
from src.evaluation.metrics import calculate_detection_metrics, calculate_map
from src.inference.predictor import MODEL_NAMES, clear_model_cache, predict


def apply_darken(image: Image.Image, factor: float = 0.5) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_blur(image: Image.Image, radius: float = 3.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def apply_gaussian_noise(
    image: Image.Image, sigma: float = 20.0, seed: int = 42
) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(image, dtype=np.float32, copy=True)
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_jpeg(image: Image.Image, quality: int = 30) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def apply_box_occlusion(
    image: Image.Image,
    gt_boxes: List[List[float]],
    area_fraction: float = 0.25,
) -> Image.Image:
    """
    Перекрывает долю area_fraction (например 0.10, 0.25, 0.40) площади каждой рамки gt_box.
    Сохраняет пропорции исходной рамки при вычислении ширины/высоты маски.
    """
    arr = np.array(image, dtype=np.uint8, copy=True)
    arr.flags.writeable = True
    img_h, img_w = arr.shape[:2]
    scale = math.sqrt(max(0.0, min(1.0, area_fraction)))

    for box in gt_boxes:
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        ow = max(1, int(round(bw * scale)))
        oh = max(1, int(round(bh * scale)))

        ox1 = max(0, int(round(cx - ow / 2.0)))
        oy1 = max(0, int(round(cy - oh / 2.0)))
        ox2 = min(img_w, ox1 + ow)
        oy2 = min(img_h, oy1 + oh)

        arr[oy1:oy2, ox1:ox2] = 0

    return Image.fromarray(arr)


def build_variants(
    image: Image.Image,
    gt_boxes: List[List[float]],
    seed: int = 42,
) -> Dict[str, Image.Image]:
    return {
        "original": image,
        "dark_0.5": apply_darken(image, factor=0.5),
        "blur_r3": apply_blur(image, radius=3.0),
        "noise_s20": apply_gaussian_noise(image, sigma=20.0, seed=seed),
        "jpeg_q30": apply_jpeg(image, quality=30),
        "occ_box_10": apply_box_occlusion(image, gt_boxes, area_fraction=0.10),
        "occ_box_25": apply_box_occlusion(image, gt_boxes, area_fraction=0.25),
        "occ_box_40": apply_box_occlusion(image, gt_boxes, area_fraction=0.40),
    }


def evaluate_variant_predictions(
    records: List[Tuple[List[List[float]], List[List[float]], List[float]]],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict[str, float]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    ap50_vals: List[float] = []
    map50_95_vals: List[float] = []

    for gt, pb_all, ps_all in records:
        if gt or pb_all:
            m_res = calculate_map(gt, pb_all, ps_all)
            ap50_vals.append(float(m_res.get("AP50", m_res.get("ap50", 0.0))))
            map50_95_vals.append(
                float(m_res.get("mAP50_95", m_res.get("mAP50:95", m_res.get("map_50_95", 0.0))))
            )

        paired = [(float(s), b) for b, s in zip(pb_all, ps_all) if float(s) >= conf_thresh]
        paired.sort(key=lambda item: item[0], reverse=True)
        pb_thr = [p[1] for p in paired[:100]]
        ps_thr = [p[0] for p in paired[:100]]

        det = calculate_detection_metrics(gt, pb_thr, ps_thr, iou_threshold=iou_thresh)
        total_tp += int(det.get("TP", det.get("tp", 0)))
        total_fp += int(det.get("FP", det.get("fp", 0)))
        total_fn += int(det.get("FN", det.get("fn", 0)))

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    ap50 = (sum(ap50_vals) / len(ap50_vals)) if ap50_vals else 0.0
    map50_95 = (sum(map50_95_vals) / len(map50_95_vals)) if map50_95_vals else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "AP50": round(ap50, 4),
        "mAP50_95": round(map50_95, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Study #1: Robustness evaluation and D metric.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    parser.add_argument("--low-conf", type=float, default=0.001)
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    img_paths = get_split_images(args.dataset_root, args.split)
    if args.max_images is not None:
        img_paths = img_paths[: args.max_images]
    label_dir = resolve_split_dir(args.dataset_root / "labels", args.split)

    dataset_items: List[Tuple[str, List[List[float]], Dict[str, Image.Image]]] = []
    for idx, img_path in enumerate(img_paths):
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            gt_boxes = load_yolo_ground_truth(label_dir / f"{img_path.stem}.txt", w, h)
            variants = build_variants(rgb, gt_boxes, seed=args.seed + idx)
            dataset_items.append((img_path.name, gt_boxes, variants))

    variant_names = list(dataset_items[0][2].keys()) if dataset_items else []
    rows: List[Dict[str, Any]] = []
    structured: Dict[str, Any] = {
        "meta": {
            "split": args.split,
            "num_images": len(dataset_items),
            "seed": args.seed,
            "occlusion_rule": "gt_boxes kept unchanged; black mask of 10%/25%/40% of each GT box area centered on GT box.",
        },
        "models": {},
    }

    for model_name in args.models:
        print(f"=== Robustness: {model_name} ({len(dataset_items)} images) ===")
        per_variant_records: Dict[str, List[Tuple[List[List[float]], List[List[float]], List[float]]]] = {
            v: [] for v in variant_names
        }

        for _, gt_boxes, variants in dataset_items:
            for v_name, v_img in variants.items():
                res = predict(
                    v_img,
                    model_name=model_name,
                    confidence=args.low_conf,
                    checkpoint_root=args.checkpoint_root,
                )
                boxes, scores = filter_detections(
                    res["boxes"],
                    res["scores"],
                    res["labels"],
                    conf_thr=args.low_conf,
                )
                per_variant_records[v_name].append((gt_boxes, boxes, scores))

        clear_model_cache()
        gc.collect()

        orig_metrics = evaluate_variant_predictions(
            per_variant_records["original"], conf_thresh=args.conf_thresh
        )
        structured["models"][model_name] = {}

        for v_name in variant_names:
            m = evaluate_variant_predictions(
                per_variant_records[v_name], conf_thresh=args.conf_thresh
            )
            d_abs_map = round(orig_metrics["mAP50_95"] - m["mAP50_95"], 4)
            d_rel_map = (
                round((orig_metrics["mAP50_95"] - m["mAP50_95"]) / orig_metrics["mAP50_95"], 4)
                if orig_metrics["mAP50_95"] > 0
                else 0.0
            )
            d_abs_f1 = round(orig_metrics["f1"] - m["f1"], 4)
            d_rel_f1 = (
                round((orig_metrics["f1"] - m["f1"]) / orig_metrics["f1"], 4)
                if orig_metrics["f1"] > 0
                else 0.0
            )

            row = {
                "model": model_name,
                "variant": v_name,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "AP50": m["AP50"],
                "mAP50_95": m["mAP50_95"],
                "D_abs_mAP50_95": d_abs_map,
                "D_rel_mAP50_95": d_rel_map,
                "D_abs_F1": d_abs_f1,
                "D_rel_F1": d_rel_f1,
            }
            rows.append(row)
            structured["models"][model_name][v_name] = row

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "robustness_results.json"
    csv_path = args.results_dir / "robustness_results.csv"

    json_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved robustness results to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()