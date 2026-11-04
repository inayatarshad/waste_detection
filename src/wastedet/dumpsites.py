"""Reader for box-labelled dumpsite datasets in YOLO format (e.g. Roboflow
"Aerial-Dumping-Sites" v6, CC BY 4.0).

Expected layout (Roboflow "YOLO" export):
    <root>/train/images/*.jpg   <root>/train/labels/*.txt
    <root>/valid/images/*.jpg   <root>/valid/labels/*.txt
Each label line: class cx cy w h (all relative to the image size).

Important: Roboflow writes augmented copies of one original image as
    <original>_jpg.rf.<hash>.jpg
Copies of the same original must stay on the same side of any split, otherwise the
model is validated on near-copies of its training images. `group_key` recovers the original.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .imageio_utils import load_rgb, normalise, resize_square

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_RF = re.compile(r"\.rf\.[0-9a-fA-F]+$")
_EXT_TAG = re.compile(r"_(jpe?g|png|tiff?|bmp)$", re.IGNORECASE)


def group_key(path) -> str:
    stem = Path(path).stem
    stem = _RF.sub("", stem)
    stem = _EXT_TAG.sub("", stem)
    return stem


@dataclass
class BoxRecord:
    image: Path
    boxes: np.ndarray   # (N, 4) float32, relative xyxy in [0, 1]
    group: str


def read_yolo_labels(path, n_classes_keep=None) -> np.ndarray:
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            if n_classes_keep is not None and c not in n_classes_keep:
                continue
            vals = [float(v) for v in parts[1:]]
            if len(vals) > 4:  # polygon label (segmentation export) -> its bounding box
                xs, ys = vals[0::2], vals[1::2]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            else:
                cx, cy, w, h = vals
                x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            rows.append([max(0, x0), max(0, y0), min(1, x1), min(1, y1)])
    b = np.array(rows, dtype=np.float32).reshape(-1, 4)
    keep = (b[:, 2] - b[:, 0] > 1e-4) & (b[:, 3] - b[:, 1] > 1e-4)
    return b[keep]


def load_split(root, split: str) -> list[BoxRecord]:
    img_dir = Path(root) / split / "images"
    lab_dir = Path(root) / split / "labels"
    if not img_dir.exists():
        return []
    recs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        recs.append(BoxRecord(p, read_yolo_labels(lab_dir / (p.stem + ".txt")), group_key(p)))
    return recs


def grouped_split(records: list[BoxRecord], val_frac: float = 0.15, seed: int = 0):
    groups = sorted({r.group for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_g = set(groups[:n_val])
    return [r for r in records if r.group not in val_g], [r for r in records if r.group in val_g]


def drop_overlap(train: list[BoxRecord], test: list[BoxRecord]):
    """Remove training images whose original also appears in the test split."""
    test_g = {r.group for r in test}
    kept = [r for r in train if r.group not in test_g]
    return kept, len(train) - len(kept)


def _aug_boxes(img, boxes, rng):
    """Flips and 90-degree rotations applied to image and relative xyxy boxes together."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
        boxes = np.stack([1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], 1)
    if rng.random() < 0.5:
        img = img[::-1, :]
        boxes = np.stack([boxes[:, 0], 1 - boxes[:, 3], boxes[:, 2], 1 - boxes[:, 1]], 1)
    for _ in range(int(rng.integers(4))):  # rotate 90 deg counter-clockwise
        img = np.rot90(img)
        boxes = np.stack([boxes[:, 1], 1 - boxes[:, 2], boxes[:, 3], 1 - boxes[:, 0]], 1)
    x = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    return np.clip(x, 0, 255).astype(np.uint8), boxes.astype(np.float32)


class DumpsiteDetDataset(Dataset):
    """torchvision-detection style samples: (image float CHW in [0,1], target dict)."""

    def __init__(self, records: list[BoxRecord], size: int = 800, train: bool = False, seed: int = 0):
        self.records, self.size, self.train, self.seed, self.epoch = records, size, train, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = resize_square(load_rgb(r.image), self.size)
        boxes = r.boxes.copy()
        if self.train:
            img, boxes = _aug_boxes(img, boxes, np.random.default_rng((self.seed, self.epoch, i)))
        img_t = torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1).astype(np.float32) / 255.0)
        b = torch.from_numpy(boxes * self.size).float().reshape(-1, 4)
        target = {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64), "image_id": torch.tensor(i)}
        return img_t, target


def tiles_for_classifier(records: list[BoxRecord], size: int = 640):
    """Image-level labels from box datasets (an image with any box = waste present).
    Used only for the optional extra training data of the scene classifier."""
    return [(r.image, int(len(r.boxes) > 0)) for r in records]


def collate_det(batch):
    return tuple(zip(*batch))


def to_classifier_tensor(path, size=640):
    return torch.from_numpy(normalise(resize_square(load_rgb(path), size)))
