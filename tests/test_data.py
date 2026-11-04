import numpy as np
import pytest
import torch

from wastedet import aerialwaste as AW
from wastedet import dumpsites as DS
from wastedet.imageio_utils import dhash, hamming


def test_records_types_and_masks(aw_root):
    recs = AW.load_records(aw_root / "training.json")
    assert len(recs) == 80
    for r in recs:
        if r.waste == 0:          # random negative location: every type is known to be absent
            assert r.fine and r.types.sum() == 0
        if r.waste == 1 and not r.fine:   # positive without detailed labels: types unknown
            assert r.types.sum() == 0
            assert AW.type_mask_of(r).sum() == 0
    assert AW.DEFAULT_TYPES[0] == "Plastic"
    assert any(r.types[0] == 1 for r in recs)


def test_type_name_matching():
    cats = [{"id": 5, "name": "Plastic"}, {"id": 7, "name": "Fire wood"}, {"id": 1, "name": "Rubble/excavated earth and rocks"}]
    assert AW.resolve_type_ids(cats, ["plastic", "Fire Wood", "Rubble/excavated earth and rocks"]) == [5, 7, 1]
    with pytest.raises(KeyError):
        AW.resolve_type_ids(cats, ["glass"])


def test_stratified_split_disjoint_and_keeps_plastic(aw_root):
    recs = AW.load_records(aw_root / "training.json")
    tr, va = AW.stratified_split(recs, 0.2, seed=1)
    assert {r.file_name for r in tr}.isdisjoint({r.file_name for r in va})
    assert len(tr) + len(va) == len(recs)
    has_pl = lambda rs: any(r.fine and r.types[0] > 0 for r in rs)
    assert has_pl(tr) and has_pl(va)


def test_scene_dataset_shapes_and_deterministic_aug(aw_root):
    recs = AW.load_records(aw_root / "training.json")
    ds = AW.WasteSceneDataset(recs, aw_root / "images", size=64, train=True)
    x, yw, yt, mk = ds[3]
    assert x.shape == (3, 64, 64) and yt.shape == (7,) and mk.shape == (7,)
    assert torch.equal(ds[3][0], x)               # same epoch -> same augmentation
    ds.epoch = 1
    assert not torch.equal(ds[3][0], x)           # new epoch -> new augmentation


def test_group_key_strips_roboflow_suffixes():
    assert DS.group_key("site12_jpg.rf.3fa9c0de12.jpg") == "site12"
    assert DS.group_key("a_b_png.rf.ABCDEF.png") == "a_b"
    assert DS.group_key("plain.jpg") == "plain"


def test_yolo_reader_and_grouped_split(yolo_root):
    tr = DS.load_split(yolo_root, "train")
    assert len(tr) == 36 and all(len(r.boxes) == 1 for r in tr)
    a, b = DS.grouped_split(tr, 0.25, seed=0)
    assert {r.group for r in a}.isdisjoint({r.group for r in b})
    kept, dropped = DS.drop_overlap(tr, tr[:3])
    assert dropped == 3 and all(r.group != tr[0].group for r in kept)


def test_polygon_labels_become_boxes(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n0 0.5 0.5 0.2 0.2\n")
    b = DS.read_yolo_labels(p)
    np.testing.assert_allclose(b, [[0.1, 0.2, 0.4, 0.6], [0.4, 0.4, 0.6, 0.6]], atol=1e-6)


@pytest.mark.parametrize("seed", range(12))
def test_box_augmentation_follows_the_image(seed):
    img = np.zeros((100, 100, 3), np.uint8)
    img[10:30, 50:90] = 255                        # object at x 50-90, y 10-30
    box = np.array([[0.5, 0.1, 0.9, 0.3]], np.float32)
    out, b = DS._aug_boxes(img, box, np.random.default_rng(seed))
    ys, xs = np.where(out.mean(2) > 200)
    x0, y0, x1, y1 = b[0] * 100
    assert xs.min() >= x0 - 1 and xs.max() <= x1 + 1 and ys.min() >= y0 - 1 and ys.max() <= y1 + 1
    assert (x1 - x0) * (y1 - y0) == pytest.approx(800, rel=0.05)


def test_dhash_finds_near_duplicates():
    rng = np.random.default_rng(0)
    a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    b = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    c = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    assert hamming(dhash(a), dhash(b)) <= 4 < hamming(dhash(a), dhash(c))
