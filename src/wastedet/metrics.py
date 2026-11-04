"""Metrics. Implemented here (and unit-tested) so every model and baseline is scored by
exactly the same code."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- scene classification
def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, same as sklearn's AP)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    # treat tied scores as one block
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    prec = tp / (tp + fp)
    rec = tp / n_pos
    rec_prev = np.r_[0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def binary_at(y, s, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    p = (np.asarray(s) >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / max(1, len(y)), "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y, s, metric: str = "f1", grid=None) -> dict:
    """Pick the decision threshold on VALIDATION data (never on test)."""
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    best = None
    for t in grid:
        m = binary_at(y, s, t)
        if best is None or m[metric] > best[metric] + 1e-12:
            best = m
    return best


def classification_report(y_waste, s_waste, y_types, s_types, type_mask, type_names, thr: float,
                          type_thr: dict | None = None) -> dict:
    rep = {"waste": {"ap": average_precision(y_waste, s_waste), **binary_at(y_waste, s_waste, thr)},
           "types": {}}
    M_ = np.asarray(type_mask).astype(bool)
    if M_.ndim == 1:
        M_ = np.repeat(M_[:, None], len(type_names), axis=1)
    for j, name in enumerate(type_names):
        m = M_[:, j]
        yt, st = np.asarray(y_types)[m, j], np.asarray(s_types)[m, j]
        # "ap": over all scenes with known types (waste sites + no-waste scenes): "is there X here?"
        # "ap_within_waste": only among waste sites: "which kind of waste is this?" (harder)
        mw = m & (np.asarray(y_waste) == 1)
        yw_, sw_ = np.asarray(y_types)[mw, j], np.asarray(s_types)[mw, j]
        entry = {"ap": average_precision(yt, st), "positives": int(yt.sum()), "images": int(m.sum()),
                 "chance_ap": float(yt.mean()) if len(yt) else float("nan"),
                 "ap_within_waste": average_precision(yw_, sw_),
                 "chance_ap_within_waste": float(yw_.mean()) if len(yw_) else float("nan")}
        if type_thr and name in type_thr:
            entry.update(binary_at(yt, st, type_thr[name]))
        rep["types"][name] = entry
    aps = [v["ap"] for v in rep["types"].values() if not np.isnan(v["ap"])]
    rep["types_mean_ap"] = float(np.mean(aps)) if aps else float("nan")
    return rep


def bootstrap_ci(y, s, fn=None, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Confidence interval of a score by resampling images. Small local test sets (Almeria) give
    noisy numbers; the interval shows how much a difference can be trusted."""
    fn = fn or average_precision
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        if y[k].min() == y[k].max():
            continue
        vals.append(fn(y[k], s[k]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if vals else (np.nan, np.nan)
    return float(lo), float(hi)


# ---------------------------------------------------------------- object detection
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def _match(preds, gts, iou_thr):
    """Greedy matching per image, highest score first. Returns (scores, is_tp, n_gt)."""
    scores, tps, n_gt = [], [], 0
    for (pb, ps), gb in zip(preds, gts):
        gb = np.asarray(gb).reshape(-1, 4); pb = np.asarray(pb).reshape(-1, 4); ps = np.asarray(ps).reshape(-1)
        n_gt += len(gb)
        order = np.argsort(-ps, kind="mergesort")
        used = np.zeros(len(gb), bool)
        ious = box_iou(pb, gb) if len(gb) and len(pb) else np.zeros((len(pb), len(gb)))
        for k in order:
            scores.append(ps[k])
            if len(gb):
                cand = np.where(~used)[0]
                if len(cand):
                    j = cand[np.argmax(ious[k, cand])]
                    if ious[k, j] >= iou_thr:
                        used[j] = True
                        tps.append(1)
                        continue
            tps.append(0)
    return np.array(scores), np.array(tps), n_gt


def detection_ap(preds, gts, iou_thr: float = 0.5) -> float:
    """preds: list per image of (boxes (N,4), scores (N,)); gts: list per image of boxes (M,4).
    All-point interpolated AP (precision envelope), as in VOC2010+/COCO-style mAP50."""
    scores, tps, n_gt = _match(preds, gts, iou_thr)
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    o = np.argsort(-scores, kind="mergesort")
    tp = np.cumsum(tps[o]); fp = np.cumsum(1 - tps[o])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[1, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def detection_map50_95(preds, gts) -> float:
    return float(np.mean([detection_ap(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)]))


def detection_pr_at(preds, gts, score_thr: float, iou_thr: float = 0.5) -> dict:
    kept = [(np.asarray(b).reshape(-1, 4)[np.asarray(s) >= score_thr], np.asarray(s)[np.asarray(s) >= score_thr])
            for b, s in preds]
    _, tps, n_gt = _match(kept, gts, iou_thr)
    tp = int(tps.sum()); fp = int(len(tps) - tp)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"score_threshold": float(score_thr), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "fp": fp, "fn": n_gt - tp}


def best_det_threshold(preds, gts, grid=None) -> dict:
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    return max((detection_pr_at(preds, gts, t) for t in grid), key=lambda m: m["f1"])
