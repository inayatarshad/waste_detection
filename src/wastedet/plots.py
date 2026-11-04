"""Figures for reports and slides (matplotlib, no Colab UI in them)."""
from __future__ import annotations

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def pr_curve(y, s):
    y = np.asarray(y).astype(int); o = np.argsort(-np.asarray(s), kind="mergesort")
    tp = np.cumsum(y[o]); fp = np.cumsum(1 - y[o])
    return tp / max(1, y.sum()), tp / np.maximum(tp + fp, 1)


def plot_pr(curves: dict, path, title="Waste / no waste (test set)", ref_point=None):
    """curves: name -> (y, scores, AP). ref_point: (recall, precision, label) e.g. a published result."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (y, s, ap) in curves.items():
        r, p = pr_curve(y, s)
        ax.plot(r, p, label=f"{name} (AP {ap:.3f})", lw=2)
    if ref_point:
        ax.scatter([ref_point[0]], [ref_point[1]], color="k", zorder=5, label=ref_point[2])
    ax.set_xlabel("Recall (share of real waste sites found)"); ax.set_ylabel("Precision (share of alarms that are real)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_history(history: list[dict], path, keys=("val_ap", "val_plastic_ap"), title="Training"):
    plt = _plt()
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["loss"] for h in history], lw=2); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    for k in keys:
        if k in history[0]:
            ax[1].plot(ep, [h[k] for h in history], lw=2, label=k)
    ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_type_aps(rep: dict, path, title="Average precision per waste type (test set)"):
    plt = _plt()
    names = list(rep["types"]); ap = [rep["types"][n]["ap"] for n in names]
    ch = [rep["types"][n]["chance_ap"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    ax.barh(y, ap, color=["#2a9d8f" if n.lower() == "plastic" else "#8ab6d6" for n in names], label="model")
    ax.scatter(ch, y, color="k", marker="|", s=200, label="chance (random guess)")
    for yi, a, n in zip(y, ap, names):
        label = "n/a - no test examples" if np.isnan(a) else f"{a:.2f}  ({rep['types'][n]['positives']} pos.)"
        ax.text(0.01 if np.isnan(a) else a + 0.03, yi, label, va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 1.15); ax.invert_yaxis()
    ax.set_xlabel("Average precision"); ax.legend(fontsize=8, loc="lower right"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def grid_with_cam(images, cams, titles, path, ncols=4):
    plt = _plt()
    n = len(images); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, cam, t in zip(np.atleast_1d(axes).ravel(), images, cams, titles):
        ax.imshow(im)
        if cam is not None:
            ax.imshow(cam, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(t, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def draw_boxes(img, gt, pred, scores, thr, path, title=""):
    plt = _plt()
    import matplotlib.patches as mp

    fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(img); ax.axis("off")
    for b in gt:
        ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=2))
    for b, s in zip(pred, scores):
        if s >= thr:
            ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=1.5, ls="--"))
            ax.text(b[0], b[1] - 3, f"{s:.2f}", color="red", fontsize=7)
    ax.set_title(title + "  (green = labelled, red = model)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
