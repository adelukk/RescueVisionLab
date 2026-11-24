import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.detection_dataset import CombinedDataset, collate_fn
from src.models.factory import MODEL_NAMES, create_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(split, batch, workers, shuffle=False):
    dataset = CombinedDataset(split=split)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
    }

    # Не уничтожать workers после каждого прохода по DataLoader.
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def prepare_batch(images, targets, device):
    non_blocking = device.type == "cuda"

    images = [
        image.to(device, non_blocking=non_blocking)
        for image in images
    ]

    targets = [
        {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in target.items()
        }
        for target in targets
    ]

    return images, targets


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    amp_enabled=False,
    scaler=None,
):
    model.train()

    # Во время validation замораживаем статистику BatchNorm.
    if not training:
        for module in model.modules():
            if isinstance(
                module,
                torch.nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()

    total_loss = 0.0
    total_images = 0

    for step, (images, targets) in enumerate(loader, 1):
        images, targets = prepare_batch(
            images, targets, device
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                losses = model(images, targets)
                loss = sum(losses.values())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at batch {step}: "
                    f"{loss.item()}"
                )

            if training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = len(images)
        total_loss += loss.detach().item() * batch_size
        total_images += batch_size

        if (
            step == 1
            or step % 25 == 0
            or step == len(loader)
        ):
            stage = "TRAIN" if training else "VAL"
            print(
                f"{stage} {step}/{len(loader)} "
                f"loss={loss.detach().item():.4f}",
                flush=True,
            )

    return total_loss / max(total_images, 1)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    args,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "seed": args.seed,
            "min_size": args.min_size,
            "max_size": args.max_size,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=480)
    parser.add_argument("--max-size", type=int, default=800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", default="checkpoints")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch < 1:
        parser.error(
            "epochs and batch must be positive"
        )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp_enabled = args.amp and device.type == "cuda"

    print("Device:", device, flush=True)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print(
        "AMP:",
        "ON" if amp_enabled else "OFF",
        flush=True,
    )
    print("Batch:", args.batch, flush=True)
    print("Workers:", args.workers, flush=True)

    train_loader = make_loader(
        "train",
        args.batch,
        args.workers,
        shuffle=True,
    )

    val_loader = make_loader(
        "val",
        args.batch,
        args.workers,
    )

    model = create_model(args.model)

    # SSD300 использует собственный transform 300x300.
    if args.model != "ssd300":
        model.transform.min_size = (args.min_size,)
        model.transform.max_size = args.max_size

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5,
    )

    # GradScaler используется только при CUDA AMP.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    output_dir = Path(args.output) / args.model
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    csv_path = output_dir / "history.csv"

    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(last_path)

        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["model_name"] != args.model:
            raise ValueError(
                "Checkpoint model mismatch"
            )

        if args.model != "ssd300":
            if (
                checkpoint["min_size"] != args.min_size
                or checkpoint["max_size"] != args.max_size
            ):
                raise ValueError(
                    "Image size settings changed"
                )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state"]
        )

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint[
            "best_val_loss"
        ]

        print(
            "Resuming from epoch",
            start_epoch,
            flush=True,
        )

    if start_epoch > args.epochs:
        print(
            "Requested epochs already completed."
        )
        return

    write_header = not (
        args.resume and csv_path.exists()
    )

    with csv_path.open(
        "a" if args.resume else "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "seconds",
                ]
            )

        for epoch in range(
            start_epoch,
            args.epochs + 1,
        ):
            print(
                f"\n=== {args.model} EPOCH "
                f"{epoch}/{args.epochs} ===",
                flush=True,
            )

            start_time = time.time()

            train_loss = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                True,
                amp_enabled=amp_enabled,
                scaler=scaler,
            )

            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    device,
                    False,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                )

            scheduler.step()

            seconds = (
                time.time() - start_time
            )

            lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    lr,
                    seconds,
                ]
            )

            file.flush()

            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f}, "
                f"val={val_loss:.4f}, "
                f"time={seconds:.1f}s",
                flush=True,
            )

            is_best = (
                val_loss < best_val_loss
            )

            if is_best:
                best_val_loss = val_loss

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args,
            )

            if is_best:
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    args,
                )

                print(
                    "New best checkpoint saved.",
                    flush=True,
                )

    print("\nTraining completed.")
    print("Results:", output_dir)


if __name__ == "__main__":
    main()