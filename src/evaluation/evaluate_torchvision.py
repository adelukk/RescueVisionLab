
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou

from src.data.detection_dataset import PDDDataset, collate_fn
from src.models.factory import MODEL_NAMES, create_model


def calculate_counts(pred, target, score_threshold, iou_threshold):
    keep = pred["scores"] >= score_threshold

    boxes = pred["boxes"][keep]
    scores = pred["scores"][keep]
    labels = pred["labels"][keep]

    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    order = torch.argsort(scores, descending=True)
    matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    tp = 0
    fp = 0

    for idx in order:
        box = boxes[idx:idx + 1]
        label = labels[idx]

        candidates = torch.where(
            (gt_labels == label) & (~matched)
        )[0]

        if len(candidates) == 0:
            fp += 1
            continue

        ious = box_iou(box, gt_boxes[candidates])[0]
        best_iou, best_index = ious.max(dim=0)

        if best_iou.item() >= iou_threshold:
            matched[candidates[best_index]] = True
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", required=True, choices=MODEL_NAMES
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="val"
    )
    parser.add_argument(
        "--checkpoint", default=None
    )
    parser.add_argument(
        "--score", type=float, default=0.5
    )
    parser.add_argument(
        "--iou", type=float, default=0.5
    )
    parser.add_argument(
        "--output", default="results"
    )

    args = parser.parse_args()

    if not 0 <= args.score <= 1:
        parser.error("--score must be between 0 and 1")

    if not 0 < args.iou <= 1:
        parser.error("--iou must be between 0 and 1")

    checkpoint_path = Path(
        args.checkpoint
        or f"checkpoints/{args.model}/best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["model_name"] != args.model:
        raise ValueError("Checkpoint model mismatch")

    model = create_model(args.model)

    if args.model != "ssd300":
        model.transform.min_size = (
            checkpoint.get("min_size", 480),
        )
        model.transform.max_size = checkpoint.get(
            "max_size", 800
        )

    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])

    dataset = PDDDataset(split=args.split)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    metric = MeanAveragePrecision(
    	box_format="xyxy",
    	iou_type="bbox",
    	class_metrics=True,
    	max_detection_thresholds=[1, 10, 100],
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader, 1):
            images = [image.to(device) for image in images]

            predictions = model(images)

            # Metrics are accumulated on CPU.
            preds_cpu = [
                {
                    key: value.detach().cpu()
                    for key, value in pred.items()
                    if key in ("boxes", "scores", "labels")
                }
                for pred in predictions
            ]

            targets_cpu = [
                {
                    key: value.detach().cpu()
                    for key, value in target.items()
                    if key in ("boxes", "labels")
                }
                for target in targets
            ]

            # mAP uses the full prediction score range.
            metric.update(preds_cpu, targets_cpu)

            for pred, target in zip(preds_cpu, targets_cpu):
                tp, fp, fn = calculate_counts(
                    pred,
                    target,
                    args.score,
                    args.iou,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

            if index % 25 == 0 or index == len(loader):
                print(
                    f"Evaluated {index}/{len(loader)}",
                    flush=True,
                )

    result = metric.compute()

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    values = {
        "model": args.model,
        "split": args.split,
        "checkpoint_epoch": checkpoint["epoch"],
        "map50": result["map_50"].item(),
        "map50_95": result["map"].item(),
        "map75": result["map_75"].item(),
        "precision": precision,
        "recall": recall,
        "score_threshold": args.score,
        "iou_threshold": args.iou,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    print("\n=== EVALUATION RESULTS ===")
    print("Model:", args.model)
    print("Split:", args.split)
    print(f"mAP@0.5: {values['map50']:.4f}")
    print(f"mAP@0.5:0.95: {values['map50_95']:.4f}")
    print(f"mAP@0.75: {values['map75']:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(
        f"TP={total_tp}, FP={total_fp}, FN={total_fn}"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (
        f"{args.model}_{args.split}_metrics.csv"
    )

    with output_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(values.keys())
        )
        writer.writeheader()
        writer.writerow(values)

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
