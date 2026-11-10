import json

import numpy as np
import pytest
import torch

from wastedet import aerialwaste as AW
from wastedet import dumpsites as DS
from wastedet.models import WasteClassifier, export_classifier_onnx
from wastedet.train_cls import ClsConfig, load_classifier, score_test_split, train_classifier
from wastedet.train_det import DetConfig, load_detector, score_det_test_split, train_detector

TINY = "resnet18"


def _cfg(**kw):
    base = dict(backbone=TINY, pretrained=False, size=64, epochs=6, batch_size=8, lr=1e-3, num_workers=0,
                amp=False, tta=True, types=list(AW.DEFAULT_TYPES))
    base.update(kw)
    return ClsConfig(**base)


def test_classifier_learns_and_tests_once(aw_root, tmp_path):
    torch.set_num_threads(2)
    recs = AW.load_records(aw_root / "training.json")
    tr, va = AW.stratified_split(recs, 0.25, seed=0)
    cfg = _cfg()
    train_ds = AW.WasteSceneDataset(tr, aw_root / "images", size=64, train=True)
    val_ds = AW.WasteSceneDataset(va, aw_root / "images", size=64)
    test_ds = AW.WasteSceneDataset(AW.load_records(aw_root / "testing.json"), aw_root / "images", size=64)
    res = train_classifier(cfg, train_ds, val_ds, tmp_path, device="cpu", log=lambda *a: None)
    assert (tmp_path / "best.pt").exists() and len(res["history"]) == 6
    assert 0 < res["thresholds"]["waste"] < 1 and "Plastic" in res["thresholds"]["types"]

    model, ckpt = load_classifier(tmp_path / "best.pt")
    assert ckpt["thresholds"] == res["thresholds"]
    rep, _ = score_test_split(model, test_ds, ckpt["thresholds"], cfg.types, tmp_path / "test_report.json", cfg, device="cpu")
    # the synthetic task is easy: if the pipeline is wired correctly the model must learn it
    assert rep["waste"]["ap"] > 0.85, rep["waste"]
    assert json.loads((tmp_path / "test_report.json").read_text())["waste"]["ap"] == pytest.approx(rep["waste"]["ap"])
    with pytest.raises(RuntimeError):   # the test split can only be scored once
        score_test_split(model, test_ds, ckpt["thresholds"], cfg.types, tmp_path / "test_report.json", cfg, device="cpu")


def test_cam_and_onnx_export(tmp_path):
    m = WasteClassifier(TINY, n_types=7, pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    cam = m.cam(x)
    assert cam.shape == (1, 64, 64) and float(cam.max()) <= 1.0 + 1e-6
    rel = export_classifier_onnx(m, tmp_path / "m.onnx", size=64)
    assert rel < 1e-3


def test_detector_trains_and_scores(yolo_root, tmp_path):
    torch.set_num_threads(2)
    tr_all = DS.load_split(yolo_root, "train")
    test = DS.load_split(yolo_root, "valid")
    tr_all, dropped = DS.drop_overlap(tr_all, test)
    assert dropped == 0
    tr, va = DS.grouped_split(tr_all, 0.25)
    cfg = DetConfig(size=128, epochs=1, batch_size=2, num_workers=0, amp=False, pretrained=False, max_steps_per_epoch=2)
    res = train_detector(cfg, DS.DumpsiteDetDataset(tr, 128, train=True), DS.DumpsiteDetDataset(va, 128), tmp_path,
                         device="cpu", log=lambda *a: None)
    model, ckpt = load_detector(tmp_path / "best_det.pt")
    assert "score_threshold" in ckpt
    rep, preds, gts = score_det_test_split(model, DS.DumpsiteDetDataset(test, 128), ckpt["score_threshold"],
                                    tmp_path / "det_test.json", device="cpu")
    assert rep["images"] == 5 and rep["boxes"] == 5
    assert 0.0 <= rep["map50"] <= 1.0 or np.isnan(rep["map50"])
    x, t = DS.DumpsiteDetDataset(tr, 128, train=True)[0]
    assert x.shape == (3, 128, 128) and t["boxes"].shape == (1, 4)


def test_prepare_cache_and_finetune_init(aw_root, tmp_path):
    recs = AW.load_records(aw_root / "training.json")[:12]
    idx = AW.index_images(aw_root)
    n = AW.prepare_cache(idx, tmp_path / "cache", [r.file_name for r in recs], size=48)
    assert n == 12 and AW.load_records  # cached copies exist at the training size
    from wastedet.imageio_utils import load_rgb
    assert load_rgb(tmp_path / "cache" / recs[0].file_name).shape == (48, 48, 3)
    tr, va = AW.stratified_split(recs, 0.3)
    cfg = _cfg(epochs=1, size=48)
    train_classifier(cfg, AW.WasteSceneDataset(tr, tmp_path / "cache", 48, True), AW.WasteSceneDataset(va, tmp_path / "cache", 48),
                     tmp_path / "a", device="cpu", log=lambda *a: None)
    logs = []
    cfg2 = _cfg(epochs=1, size=48, init_from=str(tmp_path / "a" / "best.pt"), lr=1e-5)
    train_classifier(cfg2, AW.WasteSceneDataset(tr, tmp_path / "cache", 48, True), AW.WasteSceneDataset(va, tmp_path / "cache", 48),
                     tmp_path / "b", device="cpu", log=logs.append)
    assert any("initialised from" in l for l in logs)


def test_classifier_resumes_after_disconnect(aw_root, tmp_path):
    """Simulate Colab dying after 2 of 4 epochs: the rerun must continue at epoch 3, not restart."""
    recs = AW.load_records(aw_root / "training.json")
    tr, va = AW.stratified_split(recs, 0.25)
    mk = lambda: (AW.WasteSceneDataset(tr, aw_root / "images", 64, True), AW.WasteSceneDataset(va, aw_root / "images", 64))
    train_classifier(_cfg(epochs=2), *mk(), tmp_path, device="cpu", log=lambda *a: None)
    first = json.loads((tmp_path / "history.json").read_text())
    logs = []
    res = train_classifier(_cfg(epochs=4), *mk(), tmp_path, device="cpu", log=logs.append)
    assert any("resumed" in l and "2 of 4" in l for l in logs)
    assert [h["epoch"] for h in res["history"]] == [1, 2, 3, 4] and res["history"][:2] == first
    _, ck = load_classifier(tmp_path / "best.pt")
    assert "thresholds" in ck and ck["best_epoch"] in (1, 2, 3, 4)
    # a finished run re-executed does no more training
    logs = []
    train_classifier(_cfg(epochs=4), *mk(), tmp_path, device="cpu", log=logs.append)
    assert not any(l.startswith("epoch") for l in logs)


def test_finalize_from_best_only(aw_root, tmp_path):
    """Run stopped after best.pt was written but before thresholds (old code / disconnect)."""
    from wastedet.train_cls import finalize_best
    recs = AW.load_records(aw_root / "training.json")
    tr, va = AW.stratified_split(recs, 0.25)
    cfg = _cfg(epochs=1)
    train_classifier(cfg, AW.WasteSceneDataset(tr, aw_root / "images", 64, True), AW.WasteSceneDataset(va, aw_root / "images", 64),
                     tmp_path, device="cpu", log=lambda *a: None)
    ck = torch.load(tmp_path / "best.pt", weights_only=False); ck.pop("thresholds"); torch.save(ck, tmp_path / "best.pt")
    r = finalize_best(cfg, AW.WasteSceneDataset(va, aw_root / "images", 64), tmp_path, "cpu", log=lambda *a: None)
    assert "waste" in r["thresholds"] and r["best_epoch"] == 1


def test_detector_resumes(yolo_root, tmp_path):
    tr = DS.load_split(yolo_root, "train"); va = DS.load_split(yolo_root, "valid")
    mk = lambda e: DetConfig(size=128, epochs=e, batch_size=2, num_workers=0, amp=False, pretrained=False, max_steps_per_epoch=1)
    train_detector(mk(1), DS.DumpsiteDetDataset(tr, 128, True), DS.DumpsiteDetDataset(va, 128), tmp_path, "cpu", log=lambda *a: None)
    logs = []
    r = train_detector(mk(2), DS.DumpsiteDetDataset(tr, 128, True), DS.DumpsiteDetDataset(va, 128), tmp_path, "cpu", log=logs.append)
    assert any("resumed" in l for l in logs) and len(r["history"]) == 2


def test_finished_run_reuses_saved_thresholds(aw_root, tmp_path):
    """Re-running a finished training (e.g. on a CPU-only runtime) must not redo the validation pass."""
    recs = AW.load_records(aw_root / "training.json")
    tr, va = AW.stratified_split(recs, 0.25)
    mk = lambda: (AW.WasteSceneDataset(tr, aw_root / "images", 64, True), AW.WasteSceneDataset(va, aw_root / "images", 64))
    first = train_classifier(_cfg(epochs=2), *mk(), tmp_path, device="cpu", log=lambda *a: None)
    logs = []
    again = train_classifier(_cfg(epochs=2), *mk(), tmp_path, device="cpu", log=logs.append)
    assert any("saved thresholds" in l for l in logs) and again["val_pred"] is None
    assert again["thresholds"] == first["thresholds"]
