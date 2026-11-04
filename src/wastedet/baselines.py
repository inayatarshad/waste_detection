"""Classic (non-deep-learning) baselines, scored on the same test images as the deep models.

1. Hand-crafted features + logistic regression - the standard pre-deep-learning approach for
   scene classification: colour histograms, edge density and texture statistics.
2. "White-plastic" rule for the plastic type - fraction of bright, unsaturated pixels
   (plastic film and sheeting look white/grey from above), threshold tuned on validation.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

FEAT_SIZE = 256
WHITE_COL = 40   # column of handcrafted_features() holding white_fraction (plastic rule)


def _hsv(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img).convert("HSV"), dtype=np.float32) / 255.0


def white_fraction(img: np.ndarray, v_min: float = 0.75, s_max: float = 0.18) -> float:
    hsv = _hsv(img)
    return float(((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).mean())


def handcrafted_features(img: np.ndarray) -> np.ndarray:
    img = np.asarray(Image.fromarray(img).resize((FEAT_SIZE, FEAT_SIZE), Image.BILINEAR))
    hsv = _hsv(img)
    feats = []
    for c, bins in ((0, 12), (1, 8), (2, 8)):
        h, _ = np.histogram(hsv[..., c], bins=bins, range=(0, 1))
        feats.append(h / h.sum())
    g = img.astype(np.float32).mean(2) / 255.0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]; gy[1:-1] = g[2:] - g[:-2]
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    oh, _ = np.histogram(ang[mag > 0.05], bins=8, range=(-np.pi, np.pi))
    oh = oh / max(1, oh.sum())
    ent = -np.sum(oh[oh > 0] * np.log(oh[oh > 0]))
    k = 8
    blocks = g[: FEAT_SIZE // k * k, : FEAT_SIZE // k * k].reshape(FEAT_SIZE // k, k, FEAT_SIZE // k, k)
    local_std = blocks.std(axis=(1, 3))
    feats.append(np.array([
        mag.mean(), mag.std(), (mag > 0.1).mean(), (mag > 0.2).mean(), ent,
        local_std.mean(), local_std.std(), np.percentile(local_std, 90),
        g.mean(), g.std(), hsv[..., 1].mean(), hsv[..., 1].std(),
        white_fraction(img), white_fraction(img, 0.85, 0.12),
        *img.reshape(-1, 3).astype(np.float32).mean(0) / 255.0,
        *img.reshape(-1, 3).astype(np.float32).std(0) / 255.0,
    ], dtype=np.float32))
    return np.concatenate([f.astype(np.float32) for f in feats])


class HandcraftedBaseline:
    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(C=C, class_weight="balanced", max_iter=2000))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
