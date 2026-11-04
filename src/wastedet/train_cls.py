"""Training / evaluation of the waste scene classifier.

Protocol (same rules as the GSTP models):
  * model selection (best epoch) and every decision threshold are chosen on VALIDATION data;
  * the test split is scored exactly once, at the end, with those frozen choices;
  * a classic baseline is scored on the same test images.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .models import WasteClassifier, multitask_loss


@dataclass
class ClsConfig:
    backbone: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    size: int = 640
    epochs: int = 20
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    type_weight: float = 0.5
    num_workers: int = 2
    amp: bool = True
    tta: bool = True
    seed: int = 0
    max_steps_per_epoch: int | None = None   # for quick smoke tests
    init_from: str | None = None             # start from an earlier checkpoint (fine-tuning)
    types: list = field(default_factory=list)


def set_seed(seed: int):
    import random

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _tta_views(x):
    yield x
    yield torch.flip(x, dims=[3])
    yield torch.flip(x, dims=[2])
    yield torch.rot90(x, 1, dims=[2, 3])


@torch.no_grad()
def predict(model, dataset, batch_size=32, device="cpu", tta=False, num_workers=0, amp=False):
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    S, YW, YT, MK = [], [], [], []
    for x, yw, yt, mk in dl:
        x = x.to(device, non_blocking=True)
        views = list(_tta_views(x)) if tta else [x]
        prob = 0
        for v in views:
            with torch.autocast(device_type="cuda", enabled=amp and device != "cpu"):
                prob = prob + torch.sigmoid(model(v).float())
        S.append((prob / len(views)).cpu().numpy())
        YW.append(yw.numpy()); YT.append(yt.numpy()); MK.append(mk.numpy())
    S = np.concatenate(S)
    return {"s_waste": S[:, 0], "s_types": S[:, 1:], "y_waste": np.concatenate(YW),
            "y_types": np.concatenate(YT), "mask": np.concatenate(MK)}


def _mask2d(mask, n):
    m = np.asarray(mask).astype(bool)
    return np.repeat(m[:, None], n, axis=1) if m.ndim == 1 else m


def _type_pos_weight(records, n_types):
    from .aerialwaste import type_mask_of

    masks = np.array([type_mask_of(r) for r in records]).reshape(-1, n_types)
    ys = np.array([r.types for r in records]).reshape(-1, n_types)
    pos = (ys * masks).sum(0)
    neg = masks.sum(0) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 10.0), dtype=torch.float32)


def choose_thresholds(val_pred, type_names) -> dict:
    thr = {"waste": M.best_threshold(val_pred["y_waste"], val_pred["s_waste"])["threshold"], "types": {}}
    m = _mask2d(val_pred["mask"], len(type_names))
    for j, n in enumerate(type_names):
        yt = val_pred["y_types"][m[:, j], j]
        thr["types"][n] = M.best_threshold(yt, val_pred["s_types"][m[:, j], j])["threshold"] if yt.sum() else 0.5
    return thr


def report(pred, type_names, thr) -> dict:
    return M.classification_report(pred["y_waste"], pred["s_waste"], pred["y_types"], pred["s_types"],
                                   pred["mask"], type_names, thr["waste"], thr["types"])


def train_classifier(cfg: ClsConfig, train_ds, val_ds, out_dir, device=None, log=print) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    types = cfg.types
    model = WasteClassifier(cfg.backbone, n_types=len(types), pretrained=cfg.pretrained and not cfg.init_from).to(device)
    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=device, weights_only=False)["model"])
        log(f"initialised from {cfg.init_from}")

    recs = train_ds.records
    n_pos = sum(r.waste for r in recs); n_neg = len(recs) - n_pos
    pos_weight = n_neg / max(1, n_pos)
    type_pw = _type_pos_weight(recs, len(types))
    log(f"train {len(recs)} images ({n_pos} waste / {n_neg} no waste), val {len(val_ds)}; "
        f"pos_weight {pos_weight:.2f}; type pos weights {np.round(type_pw.numpy(), 1).tolist()}")

    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, pin_memory=device != "cpu")
    steps_per_epoch = len(dl) if cfg.max_steps_per_epoch is None else min(len(dl), cfg.max_steps_per_epoch)
    total = max(1, steps_per_epoch * cfg.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(total * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device != "cpu")

    history, best = [], {"score": -1.0}
    for epoch in range(cfg.epochs):
        model.train(); train_ds.epoch = epoch
        t0, losses = time.time(), []
        for step, (x, yw, yt, mk) in enumerate(dl):
            if step >= steps_per_epoch:
                break
            x, yw, yt, mk = (t.to(device, non_blocking=True) for t in (x, yw, yt, mk))
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device != "cpu"):
                logits = model(x)
            loss, _ = multitask_loss(logits.float(), yw, yt, mk, pos_weight, type_pw, cfg.type_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        vp = predict(model, val_ds, cfg.batch_size * 2, device, tta=False, num_workers=cfg.num_workers, amp=cfg.amp)
        ap = M.average_precision(vp["y_waste"], vp["s_waste"])
        m = _mask2d(vp["mask"], len(types))
        type_aps = [M.average_precision(vp["y_types"][m[:, j], j], vp["s_types"][m[:, j], j]) for j in range(len(types))]
        plastic_ap = type_aps[0] if types and types[0].lower() == "plastic" else float("nan")
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap": ap,
               "val_types_map": float(np.nanmean(type_aps)) if type_aps else float("nan"),
               "val_plastic_ap": plastic_ap, "minutes": (time.time() - t0) / 60}
        history.append(row)
        log(f"epoch {row['epoch']:2d}  loss {row['loss']:.4f}  val AP {ap:.4f}  "
            f"types mAP {row['val_types_map']:.3f}  plastic AP {plastic_ap:.3f}  ({row['minutes']:.1f} min)")
        if ap > best["score"]:
            best = {"score": ap, "epoch": epoch + 1}
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "types": types}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    vp = predict(model, val_ds, cfg.batch_size * 2, device, tta=cfg.tta, num_workers=cfg.num_workers, amp=cfg.amp)
    thr = choose_thresholds(vp, types)
    ckpt["thresholds"] = thr; ckpt["best_epoch"] = best["epoch"]
    torch.save(ckpt, out / "best.pt")
    log(f"best epoch {best['epoch']}; thresholds chosen on validation: waste {thr['waste']:.2f}")
    return {"model": model, "history": history, "thresholds": thr, "best_epoch": best["epoch"], "val_pred": vp}


def load_classifier(path, device="cpu", pretrained=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = WasteClassifier(cfg["backbone"], n_types=len(ckpt["types"]), pretrained=pretrained)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def score_test_split(model, test_ds, thresholds, types, out_path, cfg: ClsConfig, device=None, extra: dict | None = None):
    """Score the held-out test split ONCE with the thresholds frozen on validation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} exists: the test split has already been scored. "
                           "Delete it only if you really intend to re-test.")
    tp = predict(model, test_ds, cfg.batch_size * 2, device, tta=cfg.tta, num_workers=cfg.num_workers, amp=cfg.amp)
    rep = report(tp, types, thresholds)
    srcs = np.array([r.source for r in test_ds.records])
    if len(set(srcs)) > 1:
        rep["by_source"] = {}
        for s in sorted(set(srcs)):
            k = srcs == s
            rep["by_source"][s] = {"images": int(k.sum()),
                                   "ap": M.average_precision(tp["y_waste"][k], tp["s_waste"][k]),
                                   **M.binary_at(tp["y_waste"][k], tp["s_waste"][k], thresholds["waste"])}
    rep["thresholds"] = thresholds
    if extra:
        rep.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2, default=float))
    np.savez_compressed(out_path.with_suffix(".npz"), **tp)
    return rep, tp
