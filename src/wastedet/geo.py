"""Geography helpers for Almeria: coordinates, the PNOA orthophoto WMS, scan grids and
spatial (block) splits.

Almeria lies in UTM zone 30N; the Spanish national CRS for it is ETRS89 / UTM 30N (EPSG:25830),
so every distance below is in metres.

Imagery source: PNOA "maxima actualidad" orthophotos from the Spanish IGN (25-50 cm, RGB),
free under CC BY 4.0 (credit: "PNOA cedido por (c) Instituto Geografico Nacional"). Any other
overhead RGB source of similar resolution (Google Earth export, WorldView, Pleiades Neo, drone
mosaic) can be scanned the same way through `scan.scan_image`.
"""
from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTM = "EPSG:25830"
WGS84 = "EPSG:4326"
PNOA_WMS = "https://www.ign.es/wms-inspire/pnoa-ma"
PNOA_LAYER = "OI.OrthoimageCoverage"
PNOA_CREDIT = "PNOA cedido por (c) Instituto Geografico Nacional de Espana"


def _transformer(src, dst):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def lonlat_to_utm(lon, lat):
    return _transformer(WGS84, UTM).transform(lon, lat)


def utm_to_lonlat(x, y):
    return _transformer(UTM, WGS84).transform(x, y)


@dataclass
class Chunk:
    """One WMS request: a square of ground, fetched as one image."""
    x0: float          # left (m, EPSG:25830)
    y1: float          # top  (m)
    size_m: float
    gsd: float         # metres per pixel

    @property
    def bbox(self):
        return (self.x0, self.y1 - self.size_m, self.x0 + self.size_m, self.y1)

    @property
    def pixels(self) -> int:
        return int(round(self.size_m / self.gsd))

    @property
    def key(self) -> str:
        return f"{int(self.x0)}_{int(self.y1)}_{int(self.size_m)}_{self.gsd:.2f}"


def scan_chunks(bbox_lonlat, window_m=210.0, stride_m=105.0, windows_per_side=8, gsd=0.3) -> list[Chunk]:
    """Cover an area (lon_min, lat_min, lon_max, lat_max) with chunks such that the sliding
    windows of all chunks form one seamless global grid (no window is missed or repeated)."""
    lon0, lat0, lon1, lat1 = bbox_lonlat
    xs, ys = zip(*[lonlat_to_utm(lo, la) for lo, la in ((lon0, lat0), (lon1, lat1), (lon0, lat1), (lon1, lat0))])
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xmin = np.floor(xmin / stride_m) * stride_m
    ymax = np.ceil(ymax / stride_m) * stride_m
    step = windows_per_side * stride_m
    size = step - stride_m + window_m
    chunks = []
    y = ymax
    while y - (window_m - stride_m) > ymin:
        x = xmin
        while x + (window_m - stride_m) < xmax:
            chunks.append(Chunk(float(x), float(y), float(size), gsd))
            x += step
        y -= step
    return chunks


def fetch_wms(chunk: Chunk, cache_dir=None, url=PNOA_WMS, layer=PNOA_LAYER, fmt="image/jpeg",
              retries=4, timeout=60, session=None) -> np.ndarray:
    """GetMap for one chunk (WMS 1.3.0, EPSG:25830 -> axis order x,y). Cached on disk."""
    from PIL import Image

    if cache_dir is not None:
        p = Path(cache_dir) / f"{chunk.key}.jpg"
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    import requests

    s = session or requests.Session()
    x0, y0, x1, y1 = chunk.bbox
    params = {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer, "STYLES": "",
              "CRS": UTM, "BBOX": f"{x0},{y0},{x1},{y1}", "WIDTH": chunk.pixels, "HEIGHT": chunk.pixels,
              "FORMAT": fmt}
    last = None
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
                if cache_dir is not None:
                    Path(cache_dir).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(Path(cache_dir) / f"{chunk.key}.jpg", quality=95)
                return img
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # network hiccup: back off and retry
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"WMS request failed for {chunk.key}: {last}")


def is_empty(img: np.ndarray, frac=0.5) -> bool:
    """WMS returns white/black outside coverage; skip such chunks."""
    flat = img.reshape(-1, 3)
    blank = (flat.min(1) > 250) | (flat.max(1) < 5)
    return blank.mean() > frac


def block_id(x, y, block_m=2000.0):
    return (int(np.floor(x / block_m)), int(np.floor(y / block_m)))


def block_split(x, y, block_m=2000.0, fractions=(0.6, 0.2, 0.2), seed=0) -> str:
    """Assign a location to train / val / test by the 2 km block it falls in (deterministic).
    Neighbouring windows overlap and look alike, so a random split would leak; whole blocks
    keep test areas geographically separate from training areas."""
    bx, by = block_id(x, y, block_m)
    h = int(hashlib.sha1(f"{seed}:{bx}:{by}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fractions[0]:
        return "train"
    if h < fractions[0] + fractions[1]:
        return "val"
    return "test"
