"""Small image helpers shared by all datasets."""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path) -> np.ndarray:
    """Read any image file as uint8 RGB (H, W, 3)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size. Every AerialWaste image covers the same ground footprint
    (210 m x 210 m), so a square resize keeps the ground scale consistent across sources."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def normalise(img: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW, ImageNet normalisation."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Difference hash, used to find near-duplicate images across datasets/splits."""
    g = np.asarray(Image.fromarray(img).convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
