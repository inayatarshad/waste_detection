import json

import numpy as np
import pytest
import torch

from wastedet import geo, scan
from wastedet.almeria import MixedDataset, almeria_datasets, records_from_sheet
from wastedet.aerialwaste import WasteSceneDataset, load_records
from wastedet.baselines import HandcraftedBaseline, handcrafted_features, white_fraction
from wastedet.metrics import average_precision
from wastedet.models import WasteClassifier


def test_utm_roundtrip_in_almeria():
    lon, lat = -2.4637, 36.8381          # Almeria city
    x, y = geo.lonlat_to_utm(lon, lat)
    assert 540_000 < x < 560_000 and 4_070_000 < y < 4_090_000
    lo, la = geo.utm_to_lonlat(x, y)
    assert lo == pytest.approx(lon, abs=1e-7) and la == pytest.approx(lat, abs=1e-7)


def test_chunks_form_one_seamless_window_grid():
    stride, win = 105.0, 210.0
    chunks = geo.scan_chunks((-2.20, 36.88, -2.17, 36.90), win, stride, windows_per_side=8, gsd=0.5)
    assert len(chunks) > 1
    starts = set()
    for c in chunks:
        n = int(round((c.size_m - win) / stride)) + 1
        assert n == 8 and c.pixels == int(round(c.size_m / 0.5))
        for i in range(n):
            for j in range(n):
                starts.add((round(c.x0 + j * stride, 3), round(c.y1 - i * stride, 3)))
    xs = sorted({s[0] for s in starts}); ys = sorted({s[1] for s in starts})
    assert np.allclose(np.diff(xs), stride) and np.allclose(np.diff(ys), stride)   # no gaps, no repeats
    assert len(starts) == len(xs) * len(ys)


def test_block_split_is_spatial_and_deterministic():
    a = geo.block_split(551_100, 4_080_100); b = geo.block_split(551_900, 4_080_900)
    assert a == b == geo.block_split(551_100, 4_080_100)          # same 2 km block -> same split
    splits = [geo.block_split(x, y) for x in range(500_000, 600_000, 2000) for y in range(4_050_000, 4_100_000, 2000)]
    frac = {k: splits.count(k) / len(splits) for k in ("train", "val", "test")}
    assert 0.5 < frac["train"] < 0.7 and 0.12 < frac["test"] < 0.28


def test_empty_wms_tile_detection():
    assert geo.is_empty(np.full((10, 10, 3), 255, np.uint8))
    assert not geo.is_empty(np.full((10, 10, 3), 120, np.uint8))


def test_window_origins_cover_whole_image():
    o = scan.window_origins(1000, 300, 150)
    assert o[0] == 0 and o[-1] == 700 and all(b - a <= 150 for a, b in zip(o, o[1:]))
    assert scan.window_origins(200, 300, 150) == [0]


def test_scan_cluster_geojson_and_sheet(tmp_path):
    torch.manual_seed(0)
    m = WasteClassifier("resnet18", n_types=7, pretrained=False).eval()
    img = np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    wins = scan.scan_image(m, img, gsd=1.0, window_m=200, stride_m=100, size=64, tta=True)
    assert len(wins) == 25 and all(0 <= w["waste"] <= 1 and len(w["types"]) == 7 for w in wins)
    x0, y1 = 550_000.0, 4_080_000.0
    scan.add_map_coords(wins, x0, y1, 1.0)
    assert wins[0]["x"] == x0 + 100 and wins[0]["y"] == y1 - 100
    for w in wins:                       # make a 2x2 block of 'positives' + one isolated one
        w["waste"] = 0.9 if (w["row"], w["col"]) in {(0, 0), (0, 100), (100, 0), (100, 100), (400, 400)} else 0.1
    wins[0]["waste"] = 0.95
    sites = scan.cluster_windows(wins, 0.5, stride_m=100)
    assert [s["n_windows"] for s in sites] == [4, 1] and sites[0]["score"] == 0.95
    fc = scan.sites_to_geojson(sites, ["Plastic"] + ["t"] * 6, tmp_path / "s.geojson", {"Plastic": 0.5})
    assert json.loads((tmp_path / "s.geojson").read_text())["features"][0]["properties"]["site"] == 1
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and -3 < ring[0][0] < -2 and 36 < ring[0][1] < 37.5

    samples = scan.sample_for_labelling(wins, n_top=2, n_random=3, n_border=2, thr=0.5)
    assert len(samples) == 7 and len({(s["row"], s["col"]) for s in samples}) == 7
    sheet = tmp_path / "sheet.csv"
    scan.write_sheet(samples, tmp_path / "crops", sheet,
                     lambda s: img[s["row"]:s["row"] + s["win_px"], s["col"]:s["col"] + s["win_px"]])
    rows = sheet.read_text().splitlines()
    assert len(rows) == 8 and rows[0].startswith("id,file,x,y")
    # the user labels three rows: waste+plastic, waste (plastic unknown), no waste
    lines = [rows[0]] + [r.replace(",,,", f",{lab},") if k < 3 else r
                         for k, (r, lab) in enumerate(zip(rows[1:], ["1,1", "1,", "0,0"] + [""] * 4))]
    sheet.write_text("\n".join(lines) + "\n")
    labelled = scan.read_sheet(sheet)
    assert [(r["waste"], r["plastic"]) for r in labelled] == [(1, 1), (1, None), (0, 0)]
    recs = records_from_sheet(sheet, 7)
    allr = recs["train"] + recs["val"] + recs["test"]
    by_w = {(r.waste, float(r.type_mask.sum())) for r in allr}
    assert by_w == {(1, 1.0), (1, 0.0), (0, 7.0)}
    ds = almeria_datasets(sheet, tmp_path / "crops", 7, size=64)
    n = sum(len(d) for d in ds.values())
    assert n == 3 and ds[next(k for k in ds if len(ds[k]))][0][0].shape == (3, 64, 64)


def test_mixed_dataset(aw_root):
    recs = load_records(aw_root / "training.json")
    a = WasteSceneDataset(recs[:10], aw_root / "images", 64, train=True)
    b = WasteSceneDataset(recs[10:13], aw_root / "images", 64, train=True)
    mix = MixedDataset([a, b], repeats=[1, 3])
    assert len(mix) == 19 and len(mix.records) == 19 and mix.records[-1] is recs[12]
    mix.epoch = 4
    assert a.epoch == 4 and b.epoch == 4 and mix[18][0].shape == (3, 64, 64)


def test_classic_baseline_learns_synthetic_waste(aw_root):
    from wastedet.imageio_utils import load_rgb

    tr = load_records(aw_root / "training.json"); te = load_records(aw_root / "testing.json")
    X = lambda rs: np.stack([handcrafted_features(load_rgb(aw_root / "images" / r.file_name)) for r in rs])
    b = HandcraftedBaseline().fit(X(tr), np.array([r.waste for r in tr]))
    assert average_precision([r.waste for r in te], b.score(X(te))) > 0.7
    assert white_fraction(np.full((8, 8, 3), 250, np.uint8)) == 1.0
    assert white_fraction(np.full((8, 8, 3), 90, np.uint8)) == 0.0


def test_fetch_wms_builds_request_and_caches(tmp_path):
    import io

    from PIL import Image

    class Resp:
        ok, status_code, text = True, 200, ""
        headers = {"content-type": "image/jpeg"}

        def __init__(self, w):
            b = io.BytesIO(); Image.new("RGB", (w, w), (90, 120, 60)).save(b, "JPEG"); self.content = b.getvalue()

    class Session:
        calls = []

        def get(self, url, params, timeout):
            self.calls.append(params); return Resp(params["WIDTH"])

    s = Session()
    c = geo.Chunk(550_000.0, 4_080_000.0, 1050.0, 0.5)
    img = geo.fetch_wms(c, cache_dir=tmp_path, session=s)
    assert img.shape == (2100, 2100, 3)
    p = s.calls[0]
    assert p["CRS"] == "EPSG:25830" and p["BBOX"] == "550000.0,4078950.0,551050.0,4080000.0" and p["VERSION"] == "1.3.0"
    geo.fetch_wms(c, cache_dir=tmp_path, session=s)        # second time: from the disk cache
    assert len(s.calls) == 1


def test_crop_from_cached_chunk(tmp_path):
    from PIL import Image

    c = geo.Chunk(550_000.0, 4_080_000.0, 100.0, 0.5)          # 200 x 200 px chunk
    img = np.zeros((200, 200, 3), np.uint8); img[90:110, 90:110] = 255
    Image.fromarray(img).save(tmp_path / f"{c.key}.jpg", quality=100)
    crop = scan.crop_from_chunks([c], tmp_path, 550_050.0, 4_079_950.0, 20.0, 0.5)   # chunk centre
    assert crop.shape == (40, 40, 3) and crop[20, 20].mean() > 200 and crop[0, 0].mean() < 50
    assert scan.crop_from_chunks([c], tmp_path, 560_000.0, 4_079_950.0, 20.0, 0.5) is None
