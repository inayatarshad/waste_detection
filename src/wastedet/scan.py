"""Scan large overhead images / whole areas with the trained models.

The classifier was trained on 210 m x 210 m windows, so a big image is cut into 210 m windows
(50 % overlap) whatever its resolution; each window is resized to the training size. This keeps
the ground scale the model sees the same as in training, which matters more than pixel count.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .imageio_utils import normalise, resize_square


def window_origins(n_pixels: int, win_px: int, stride_px: int) -> list[int]:
    if n_pixels <= win_px:
        return [0]
    o = list(range(0, n_pixels - win_px + 1, stride_px))
    if o[-1] != n_pixels - win_px:
        o.append(n_pixels - win_px)
    return o


@torch.no_grad()
def scan_image(model, img: np.ndarray, gsd: float, window_m: float = 210.0, stride_m: float = 105.0,
               size: int = 640, batch: int = 16, device="cpu", tta: bool = False,
               exact_grid: bool = False) -> list[dict]:
    """Score every window of one image. Returns window dicts with pixel offsets (row, col),
    window size in pixels and the waste/type probabilities.
    exact_grid=True: windows only at multiples of the stride (used for seamless chunk mosaics)."""
    model.eval()
    win = int(round(window_m / gsd)); stride = int(round(stride_m / gsd))
    H, W = img.shape[:2]
    if exact_grid:
        rows = list(range(0, H - win + 1, stride)) or [0]
        cols = list(range(0, W - win + 1, stride)) or [0]
    else:
        rows, cols = window_origins(H, win, stride), window_origins(W, win, stride)
    coords = [(r, c) for r in rows for c in cols]
    out = []
    for i in range(0, len(coords), batch):
        part = coords[i:i + batch]
        x = torch.stack([torch.from_numpy(normalise(resize_square(np.ascontiguousarray(img[r:r + win, c:c + win]), size)))
                         for r, c in part]).to(device)
        views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.rot90(x, 1, [2, 3])] if tta else [x]
        p = sum(torch.sigmoid(model(v).float()) for v in views) / len(views)
        p = p.cpu().numpy()
        for (r, c), pi in zip(part, p):
            out.append({"row": r, "col": c, "win_px": win, "waste": float(pi[0]), "types": pi[1:].tolist()})
    return out


def add_map_coords(windows: list[dict], x0: float, y1: float, gsd: float) -> list[dict]:
    """Attach UTM centre coordinates for windows of an image whose top-left corner is (x0, y1)."""
    for w in windows:
        half = w["win_px"] / 2
        w["x"] = x0 + (w["col"] + half) * gsd
        w["y"] = y1 - (w["row"] + half) * gsd
        w["size_m"] = w["win_px"] * gsd
    return windows


def cluster_windows(windows: list[dict], thr: float, stride_m: float = 105.0) -> list[dict]:
    """Group overlapping/adjacent windows above threshold into candidate sites
    (connected components on the window grid)."""
    pos = [w for w in windows if w["waste"] >= thr]
    if not pos:
        return []
    keys = {(round(w["x"] / stride_m), round(w["y"] / stride_m)): w for w in pos}
    seen, sites = set(), []
    for k in keys:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            cur = stack.pop(); comp.append(keys[cur])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cur[0] + dx, cur[1] + dy)
                    if nb in keys and nb not in seen:
                        seen.add(nb); stack.append(nb)
        best = max(comp, key=lambda w: w["waste"])
        half = [w["size_m"] / 2 for w in comp]
        sites.append({
            "x": best["x"], "y": best["y"], "score": best["waste"], "n_windows": len(comp),
            "types_max": np.max([w["types"] for w in comp], axis=0).tolist() if comp[0]["types"] else [],
            "bbox_utm": (min(w["x"] - h for w, h in zip(comp, half)), min(w["y"] - h for w, h in zip(comp, half)),
                         max(w["x"] + h for w, h in zip(comp, half)), max(w["y"] + h for w, h in zip(comp, half))),
        })
    return sorted(sites, key=lambda s: -s["score"])


def sites_to_geojson(sites: list[dict], type_names: list[str], path=None, type_thr: dict | None = None) -> dict:
    from .geo import utm_to_lonlat

    feats = []
    for i, s in enumerate(sites):
        x0, y0, x1, y1 = s["bbox_utm"]
        ring = [utm_to_lonlat(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        lon, lat = utm_to_lonlat(s["x"], s["y"])
        props = {"site": i + 1, "waste_score": round(s["score"], 3), "windows": s["n_windows"],
                 "lon": round(lon, 6), "lat": round(lat, 6)}
        for n, v in zip(type_names, s["types_max"]):
            props[f"{n}_score"] = round(float(v), 3)
            if type_thr and n in type_thr:
                props[f"{n}_flag"] = bool(v >= type_thr[n])
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [[list(map(float, p)) for p in ring]]}})
    fc = {"type": "FeatureCollection", "features": feats}
    if path:
        Path(path).write_text(json.dumps(fc, indent=1))
    return fc


# ------------------------------------------------------------- labelling loop for Almeria
SHEET_FIELDS = ["id", "file", "x", "y", "lon", "lat", "split", "model_waste", "model_plastic",
                "why_sampled", "waste", "plastic", "greenhouse", "notes"]


def sample_for_labelling(windows: list[dict], n_top: int = 300, n_random: int = 300, n_border: int = 100,
                         thr: float = 0.5, seed: int = 0) -> list[dict]:
    """Which windows should a person label? Three groups, so the labels are useful AND the
    resulting test set is not biased towards what the current model already finds:
      top     - highest scores (find real dumps, and the greenhouse false alarms to fix)
      border  - scores near the threshold (most informative for the model)
      random  - uniform sample (needed for an honest false-alarm / miss rate)."""
    rng = np.random.default_rng(seed)
    order = np.argsort([-w["waste"] for w in windows])
    chosen, taken = [], set()

    def take(idx, why):
        for i in idx:
            if i not in taken:
                taken.add(i); chosen.append({**windows[i], "why_sampled": why})

    take(order[:n_top], "top")
    border = [i for i in np.argsort([abs(w["waste"] - thr) for w in windows]) if i not in taken][:n_border]
    take(border, "border")
    rest = [i for i in range(len(windows)) if i not in taken]
    take(rng.choice(rest, size=min(n_random, len(rest)), replace=False).tolist() if rest else [], "random")
    return chosen


def write_sheet(samples: list[dict], crops_dir, sheet_path, get_crop, plastic_index: int = 0, block_m=2000.0):
    """Save one JPEG per sampled window + a CSV the user fills in (waste 1/0, plastic 1/0)."""
    from PIL import Image

    from .geo import block_split, utm_to_lonlat

    crops_dir = Path(crops_dir); crops_dir.mkdir(parents=True, exist_ok=True)
    with open(sheet_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        wr.writeheader()
        for i, s in enumerate(samples):
            name = f"w_{int(s['x'])}_{int(s['y'])}.jpg"
            crop = get_crop(s)
            if crop is None or crop.size == 0:      # e.g. outside the downloaded imagery
                continue
            Image.fromarray(crop).save(crops_dir / name, quality=92)
            lon, lat = utm_to_lonlat(s["x"], s["y"])
            wr.writerow({"id": i, "file": name, "x": round(s["x"], 1), "y": round(s["y"], 1),
                         "lon": round(lon, 6), "lat": round(lat, 6), "split": block_split(s["x"], s["y"], block_m),
                         "model_waste": round(s["waste"], 3),
                         "model_plastic": round(s["types"][plastic_index], 3) if s["types"] else "",
                         "why_sampled": s["why_sampled"], "waste": "", "plastic": "", "greenhouse": "",
                         "notes": ""})


def read_sheet(sheet_path) -> list[dict]:
    """Labelled rows only. waste must be 0/1; plastic may be blank (unknown)."""
    rows = []
    with open(sheet_path, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("waste", "")).strip() not in ("0", "1"):
                continue
            p = str(r.get("plastic", "")).strip()
            rows.append({**r, "waste": int(r["waste"]),
                         "plastic": int(p) if p in ("0", "1") else (0 if r["waste"].strip() == "0" else None)})
    return rows


def crop_from_chunks(chunks, cache_dir, x: float, y: float, size_m: float, gsd: float) -> np.ndarray | None:
    """Cut a size_m x size_m square centred on (x, y) out of the cached chunk that contains it."""
    from .geo import fetch_wms

    half = size_m / 2
    for c in chunks:
        x0, y0, x1, y1 = c.bbox
        if x0 <= x <= x1 and y0 <= y <= y1:
            img = fetch_wms(c, cache_dir=cache_dir)
            g = c.gsd
            col, row = (x - x0) / g, (y1 - y) / g
            h = int(round(half / g))
            r0, c0 = int(max(0, row - h)), int(max(0, col - h))
            crop = img[r0:int(row + h), c0:int(col + h)]
            return crop if crop.size else None
    return None
