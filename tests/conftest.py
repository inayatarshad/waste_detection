"""Synthetic stand-ins for the real datasets, so every code path runs without downloads."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CATS = ["Rubble/excavated earth and rocks", "Bulky items", "Fire Wood", "Scrap", "Plastic", "Vehicles",
        "Tires", "Heaps not delimited", "Full container"]


def fake_scene(rng, waste: bool, plastic: bool, size=96):
    """Brown/green 'countryside'; waste = cluster of colourful blocks; plastic = white sheets."""
    img = np.zeros((size, size, 3), np.float32)
    img[:] = rng.uniform([110, 90, 60], [150, 130, 90])
    img += rng.normal(0, 8, img.shape)
    if waste:
        cx, cy = rng.integers(20, size - 20, 2)
        for _ in range(12):
            x, y = cx + rng.integers(-15, 15), cy + rng.integers(-15, 15)
            w, h = rng.integers(3, 8, 2)
            img[max(0, y):y + h, max(0, x):x + w] = rng.uniform(20, 235, 3)
    if plastic:
        x, y = rng.integers(5, size - 30, 2)
        img[y:y + 18, x:x + 22] = rng.uniform(235, 255)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_aerialwaste(root: Path, n_train=80, n_test=40, seed=0):
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CATS)]
    plastic_id = CATS.index("Plastic") + 1
    nid = 0
    for split, n in (("training", n_train), ("testing", n_test)):
        imgs = []
        for _ in range(n):
            waste = bool(rng.random() < 0.4)
            fine = waste and rng.random() < 0.7
            plastic = waste and rng.random() < 0.5
            name = f"img_{nid:04d}.png"
            Image.fromarray(fake_scene(rng, waste, plastic)).save(root / "images" / name)
            categories = []
            if fine:
                categories = [plastic_id] if plastic else []
                categories.append(int(rng.choice([1, 2, 4])))
            imgs.append({"id": nid, "file_name": name, "is_candidate_location": int(waste),
                         "valid_fine_grain": bool(fine), "categories": categories,
                         "width": 96, "height": 96, "evidence": 1, "severity": 1, "site_type": "x"})
            nid += 1
        (root / f"{split}.json").write_text(json.dumps({"images": imgs, "categories": cats, "annotations": []}))
    return root


def make_yolo(root: Path, n_orig=12, copies=3, n_valid=5, seed=0, size=128):
    """Roboflow-like export: train has augmented copies '<orig>_jpg.rf.<hash>.jpg'."""
    rng = np.random.default_rng(seed)
    for split, n, cp in (("train", n_orig, copies), ("valid", n_valid, 1)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for k in range(n):
            img = fake_scene(rng, False, False, size)
            x0, y0 = rng.integers(10, size - 50, 2)
            img[y0:y0 + 35, x0:x0 + 40] = rng.uniform(20, 235, 3)
            lab = f"0 {(x0 + 20) / size:.5f} {(y0 + 17.5) / size:.5f} {40 / size:.5f} {35 / size:.5f}\n"
            for c in range(cp):
                stem = f"{split}site{k}_jpg.rf.{rng.integers(1e9, 1e10):x}"
                Image.fromarray(img).save(root / split / "images" / f"{stem}.jpg", quality=95)
                (root / split / "labels" / f"{stem}.txt").write_text(lab)
    return root


@pytest.fixture(scope="session")
def aw_root(tmp_path_factory):
    return make_aerialwaste(tmp_path_factory.mktemp("aerialwaste"))


@pytest.fixture(scope="session")
def yolo_root(tmp_path_factory):
    return make_yolo(tmp_path_factory.mktemp("dumpsites"))
