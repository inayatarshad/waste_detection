"""AerialWaste reader (Torres & Fraternali, Scientific Data 2023; Zenodo record 12607190).

Layout expected after download:
    <root>/training.json, <root>/testing.json, <root>/images/<file_name>

Every image is centred on one location and covers about 210 m x 210 m of ground,
whatever its source (AGEA orthophoto ~20 cm, WorldView-3 ~30 cm, Google Earth ~50 cm).

Two label levels are used:
  * waste      - 1 if the location is a reported waste site ("is_candidate_location"), else 0
  * types      - multi-label waste-object types (Plastic, Rubble, ...) for the subset
                 that the photo interpreters annotated in detail ("valid_fine_grain").
                 For other images the type labels are unknown and are masked out of the loss.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .imageio_utils import load_rgb, normalise, resize_square

FOOTPRINT_M = 210.0  # ground size covered by one AerialWaste image

# Waste-object types the model learns as extra outputs (matched by name, case-insensitive).
DEFAULT_TYPES = [
    "Plastic",
    "Rubble/excavated earth and rocks",
    "Bulky items",
    "Scrap",
    "Tires",
    "Fire Wood",
    "Vehicles",
]


@dataclass
class Record:
    file_name: str
    waste: int
    fine: bool                      # True when the type labels below are known
    types: np.ndarray               # float32 (n_types,), zeros when fine is False
    source: str = "unknown"
    type_mask: np.ndarray | None = None   # per-type "label is known" (defaults to `fine` for all)
    meta: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("-", " ").split())


def resolve_type_ids(categories: list[dict], wanted: list[str]) -> list[int]:
    """Map wanted type names to the category ids in the json (case/spacing tolerant)."""
    by_name = {_norm(c["name"]): c["id"] for c in categories}
    ids = []
    for w in wanted:
        key = _norm(w)
        if key in by_name:
            ids.append(by_name[key])
            continue
        hits = [cid for n, cid in by_name.items() if n.startswith(key) or key.startswith(n)]
        if len(hits) != 1:
            raise KeyError(f"Waste type '{w}' not found unambiguously in {sorted(by_name)}")
        ids.append(hits[0])
    return ids


def _source_of(img: dict) -> str:
    for k in ("source", "provenance", "img_source", "dataset_source"):
        if k in img and img[k]:
            return str(img[k])
    name = img["file_name"].lower()
    for tag in ("agea", "wv3", "worldview", "google", "ge_"):
        if tag in name:
            return {"worldview": "wv3", "ge_": "google"}.get(tag, tag)
    return "unknown"


def load_records(json_path, types: list[str] = DEFAULT_TYPES) -> list[Record]:
    with open(json_path) as f:
        data = json.load(f)
    type_ids = resolve_type_ids(data["categories"], types)
    out = []
    for img in data["images"]:
        fine = bool(img.get("valid_fine_grain", img.get("is_valid_fine_grain", False)))
        cats = set(img.get("categories") or [])
        vec = np.array([1.0 if t in cats else 0.0 for t in type_ids], dtype=np.float32)
        waste = int(img["is_candidate_location"])
        if not fine:
            vec[:] = 0.0
        out.append(Record(
            file_name=img["file_name"], waste=waste, fine=fine and waste == 1, types=vec,
            source=_source_of(img),
            meta={k: img.get(k) for k in ("id", "evidence", "severity", "site_type")},
        ))
    # Negatives are random locations: they contain no waste, so all types are known to be 0.
    for r in out:
        if r.waste == 0:
            r.fine = True
            r.types[:] = 0.0
    return out


def type_mask_of(r: Record) -> np.ndarray:
    if r.type_mask is not None:
        return r.type_mask.astype(np.float32)
    return np.full(len(r.types), 1.0 if r.fine else 0.0, dtype=np.float32)


def stratified_split(records: list[Record], val_frac: float = 0.15, seed: int = 0):
    """Carve a validation set out of the official training split.
    Stratified on (waste, has-plastic) so the rare plastic class appears in both parts."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        key = (r.waste, int(r.fine and r.types[0] > 0))
        strata.setdefault(key, []).append(i)
    tr, va = [], []
    for idx in strata.values():
        idx = list(rng.permutation(idx))
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 2:
            n_val = min(max(n_val, 1), len(idx) - 1)
        va += idx[:n_val]
        tr += idx[n_val:]
    return [records[i] for i in sorted(tr)], [records[i] for i in sorted(va)]


def summarise(records: list[Record], types: list[str] = DEFAULT_TYPES) -> dict:
    fine = [r for r in records if r.fine and r.waste == 1]
    return {
        "images": len(records),
        "waste": sum(r.waste for r in records),
        "no_waste": sum(1 - r.waste for r in records),
        "waste_with_type_labels": len(fine),
        "type_counts": {t: int(sum(r.types[i] for r in fine)) for i, t in enumerate(types)},
        "sources": {s: sum(r.source == s for r in records) for s in sorted({r.source for r in records})},
    }


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overhead imagery has no 'up': flips + 90-degree rotations, a mild zoom and colour jitter
    (different flights/satellites have different colour balance)."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    img = np.rot90(img, k=int(rng.integers(4)))
    h, w = img.shape[:2]
    s = rng.uniform(0.85, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    img = resize_square(np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), h)
    x = img.astype(np.float32)
    x = x * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)               # contrast, brightness
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * rng.uniform(0.8, 1.2)                         # saturation
    x = x * rng.uniform(0.95, 1.05, size=(1, 1, 3))                       # colour balance
    return np.clip(x, 0, 255).astype(np.uint8)


class WasteSceneDataset(Dataset):
    """Returns (image CHW float, waste label, type labels, type mask)."""

    def __init__(self, records: list[Record], image_dir, size: int = 640, train: bool = False, seed: int = 0):
        self.records = records
        self.image_dir = Path(image_dir)
        self.size = size
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def image(self, i: int) -> np.ndarray:
        return resize_square(load_rgb(self.image_dir / self.records[i].file_name), self.size)

    def __getitem__(self, i):
        r = self.records[i]
        img = self.image(i)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            img = augment(img, rng)
        return (
            torch.from_numpy(normalise(img)),
            torch.tensor(float(r.waste)),
            torch.from_numpy(r.types.copy()),
            torch.from_numpy(type_mask_of(r)),
        )


def index_images(root) -> dict[str, Path]:
    """basename -> path for every image under root (zip layouts differ between versions)."""
    return {p.name: p for p in Path(root).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}}


def prepare_cache(src, dst_dir, file_names, size: int = 640, quality: int = 95) -> int:
    """Resize originals (~1000 px) to the training size once and keep them on Drive, so later
    Colab sessions do not need the 18.8 GB download again. `src` is a folder or a
    basename -> path dict from `index_images`. Returns how many images are in the cache."""
    from PIL import Image

    lookup = src if isinstance(src, dict) else None
    done = 0
    for name in file_names:
        dst = Path(dst_dir) / name
        if dst.exists():
            done += 1
            continue
        path = lookup.get(Path(name).name) if lookup is not None else Path(src) / name
        if path is None or not Path(path).exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(resize_square(load_rgb(path), size)).save(dst, quality=quality)
        done += 1
    return done
