import csv
from pathlib import Path

import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

from src.data.detection_dataset import PDDDataset


CHECKPOINT = Path(
    "runs/detect/runs/rescuevision/"
    "yolo26n_final/weights/best.pt"
)

OUTPUT_DIR = Path("results_final")

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640


def calculate_counts(pred_boxes, pred_scores, gt_boxes):
    keep = pred_scores >= SCORE_THRESHOLD

    boxes = pred_boxes[keep]
    scores = pred_scores[keep]

    order = torch.argsort(scores, descending=True)

    matched = torch.zeros(
        len(gt_boxes),
        dtype=torch.bool,
    )

    tp = 0
    fp = 0

    for idx in order:
        box = boxes[idx:idx + 1]

        candidates = torch.where(~matched)[0]

        if len(candidates) == 0:
            fp += 1
            continue

        ious = box_iou(
            box,
            gt_boxes[candidates],
        )[0]

        best_iou, best_index = ious.max(dim=0)

        if best_iou.item() >= IOU_THRESHOLD:
            matched[candidates[best_index]] = True
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp

    return tp, fp, fn


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    print("Checkpoint:", CHECKPOINT)
    print("Split: test")
    print("Score threshold:", SCORE_THRESHOLD)
    print("IoU threshold:", IOU_THRESHOLD)

    model = YOLO(str(CHECKPOINT))

    dataset = PDDDataset(split="test")

    print("Test images:", len(dataset))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index in range(len(dataset)):
        image, target = dataset[index]

        # PDDDataset returns a CHW float tensor in [0, 1].
        # Ultralytics accepts an HWC NumPy image.
        image_np = (
            image.permute(1, 2, 0)
            .mul(255)
            .clamp(0, 255)
            .byte()
            .cpu()
            .numpy()
        )

        results = model.predict(
            source=image_np,
            imgsz=IMAGE_SIZE,
            conf=SCORE_THRESHOLD,
            iou=0.7,
            max_det=100,
            device=0,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = torch.empty(
                (0, 4),
                dtype=torch.float32,
            )
            pred_scores = torch.empty(
                (0,),
                dtype=torch.float32,
            )
        else:
            pred_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .float()
            )

            pred_scores = (
                result.boxes.conf
                .detach()
                .cpu()
                .float()
            )

        gt_boxes = (
            target["boxes"]
            .detach()
            .cpu()
            .float()
        )

        tp, fp, fn = calculate_counts(
            pred_boxes,
            pred_scores,
            gt_boxes,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        current = index + 1

        if current % 25 == 0 or current == len(dataset):
            print(
                f"Evaluated {current}/{len(dataset)}",
                flush=True,
            )

    precision = total_tp / max(
        total_tp + total_fp,
        1,
    )

    recall = total_tp / max(
        total_tp + total_fn,
        1,
    )

    print("\n=== YOLO EVALUATION RESULTS ===")
    print("Model: yolo26n")
    print("Split: test")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(
        f"TP={total_tp}, "
        f"FP={total_fp}, "
        f"FN={total_fn}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR /
        "yolo26n_test_threshold_metrics.csv"
    )

    values = {
        "model": "yolo26n",
        "split": "test",
        "precision": precision,
        "recall": recall,
        "score_threshold": SCORE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(values.keys()),
        )

        writer.writeheader()
        writer.writerow(values)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()