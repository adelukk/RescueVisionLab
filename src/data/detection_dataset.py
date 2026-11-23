
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class PDDDataset(Dataset):
    def __init__(self, root="datasets/processed/yolo", split="train"):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")

        root = Path(root)
        self.images_dir = root / "images" / split
        self.labels_dir = root / "labels" / split

        if not self.images_dir.is_dir() or not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset folders: {split}")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        if not self.images:
            raise RuntimeError(f"No images found in {self.images_dir}")

        missing = [
            p.name for p in self.images
            if not (self.labels_dir / f"{p.stem}.txt").is_file()
        ]
        if missing:
            raise RuntimeError(f"Missing labels: {missing[:10]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as im:
            image = im.convert("RGB")

        width, height = image.size
        boxes = []

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            if not line.strip():
                continue

            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Invalid label: {label_path}:{line_number}")

            cls = int(values[0])
            xc, yc, bw, bh = map(float, values[1:])

            if cls != 0:
                raise ValueError(f"Unexpected class {cls}: {label_path}")

            if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                raise ValueError(f"Coordinates out of range: {label_path}")

            x1 = max(0.0, (xc - bw / 2) * width)
            y1 = max(0.0, (yc - bh / 2) * height)
            x2 = min(float(width), (xc + bw / 2) * width)
            y2 = min(float(height), (yc + bh / 2) * height)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid box: {label_path}")

            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones(len(boxes), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            ),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))
