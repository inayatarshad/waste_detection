"""Training / evaluation of the dumpsite box detector (torchvision Faster R-CNN)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import metrics as M
from .dumpsites import collate_det
from .models import build_detector


@dataclass
class DetConfig:
    size: int = 800
    epochs: int = 24
    batch_size: int = 4
    lr: float = 0.01
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    num_workers: int = 2
    amp: bool = True
    pretrained: bool = True
    seed: int = 0
    max_steps_per_epoch: int | None = None
    resume: bool = True          # continue from out_dir/last_det.pt after a Colab disconnect


@torch.no_grad()
def predict_det(model, dataset, device="cpu", batch_size=4, num_workers=0):
    """Returns (preds, gts) in pixel coordinates of the resized image."""
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_det)
    preds, gts = [], []
    for imgs, targets in dl:
        outs = model([i.to(device) for i in imgs])
        for o, t in zip(outs, targets):
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()))
            gts.append(t["boxes"].numpy())
    return preds, gts


def det_report(preds, gts, score_thr) -> dict:
    return {"map50": M.detection_ap(preds, gts, 0.5), "map50_95": M.detection_map50_95(preds, gts),
            "images": len(gts), "boxes": int(sum(len(g) for g in gts)),
            **M.detection_pr_at(preds, gts, score_thr)}


def train_detector(cfg: DetConfig, train_ds, val_ds, out_dir, device=None, log=print) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model = build_detector(cfg.pretrained, cfg.size, cfg.size).to(device)
    dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    collate_fn=collate_det, drop_last=True)
    steps = len(dl) if cfg.max_steps_per_epoch is None else min(len(dl), cfg.max_steps_per_epoch)
    total = max(1, steps * cfg.epochs); warm = max(1, int(total * cfg.warmup_frac))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device != "cpu")
    history, best, start = [], {"map50": -1.0}, 0
    last = out / "last_det.pt"
    if cfg.resume and last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        history, best, start = st["history"], st["best"], st["epoch"]
        log(f"resumed from {last}: {start} of {cfg.epochs} epochs already done")
    for epoch in range(start, cfg.epochs):
        model.train(); train_ds.epoch = epoch
        t0, losses = time.time(), []
        for step, (imgs, targets) in enumerate(dl):
            if step >= steps:
                break
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=cfg.amp and device != "cpu"):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        preds, gts = predict_det(model, val_ds, device, cfg.batch_size, cfg.num_workers)
        m50 = M.detection_ap(preds, gts, 0.5)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_map50": m50, "minutes": (time.time() - t0) / 60}
        history.append(row)
        log(f"epoch {row['epoch']:2d}  loss {row['loss']:.4f}  val mAP50 {m50:.4f}  ({row['minutes']:.1f} min)")
        if m50 > best["map50"]:
            best = {"map50": m50, "epoch": epoch + 1}
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, out / "best_det.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history, "best": best},
                   out / "last_det.pt.tmp")
        (out / "last_det.pt.tmp").replace(last)
        (out / "history_det.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(out / "best_det.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    preds, gts = predict_det(model, val_ds, device, cfg.batch_size, cfg.num_workers)
    thr = M.best_det_threshold(preds, gts)["score_threshold"]
    ckpt["score_threshold"] = thr; ckpt["best_epoch"] = best.get("epoch")
    torch.save(ckpt, out / "best_det.pt")
    log(f"best epoch {best.get('epoch')}; score threshold chosen on validation: {thr:.2f}")
    return {"model": model, "history": history, "score_threshold": thr, "best_epoch": best.get("epoch")}


def load_detector(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = build_detector(False, ckpt["config"]["size"], ckpt["config"]["size"])
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval(), ckpt


def score_det_test_split(model, test_ds, score_thr, out_path, device=None, batch_size=4, extra=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} exists: the test split has already been scored.")
    preds, gts = predict_det(model, test_ds, device, batch_size)
    rep = det_report(preds, gts, score_thr)
    if extra:
        rep.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2, default=float))
    return rep, preds, gts
