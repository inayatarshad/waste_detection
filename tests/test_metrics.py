import numpy as np
import pytest
import torch

from wastedet import metrics as M
from wastedet.models import multitask_loss


def test_ap_matches_sklearn():
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 200); s = rng.random(200)
        s[:30] = 0.5  # ties
        assert M.average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-9)


def test_threshold_is_chosen_from_given_data():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    best = M.best_threshold(y, s)
    assert 0.3 < best["threshold"] <= 0.7 and best["f1"] == 1.0
    m = M.binary_at(y, s, 0.75)
    assert (m["tp"], m["fn"], m["fp"]) == (2, 1, 0)


def test_detection_ap_cases():
    gt = [np.array([[0, 0, 10, 10]]), np.array([[20, 20, 40, 40]])]
    perfect = [(np.array([[0, 0, 10, 10]]), np.array([0.9])), (np.array([[20, 20, 40, 40]]), np.array([0.8]))]
    assert M.detection_ap(perfect, gt) == pytest.approx(1.0)
    # a confident false alarm ranked first halves precision at the start
    fa = [(np.array([[50, 50, 60, 60], [0, 0, 10, 10]]), np.array([0.99, 0.9])),
          (np.array([[20, 20, 40, 40]]), np.array([0.8]))]
    assert M.detection_ap(fa, gt) == pytest.approx(2 / 3)
    missed = [(np.array([[0, 0, 10, 10]]), np.array([0.9])), (np.zeros((0, 4)), np.zeros(0))]
    assert M.detection_ap(missed, gt) == pytest.approx(0.5)
    # a duplicate detection of the same object counts as a false alarm
    dup = [(np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), np.array([0.9, 0.85])), (np.array([[20, 20, 40, 40]]), np.array([0.8]))]
    pr = M.detection_pr_at(dup, gt, 0.5)
    assert (pr["tp"], pr["fp"], pr["fn"]) == (2, 1, 0)
    loose = [(np.array([[0, 0, 10, 13]]), np.array([0.9])), (np.array([[20, 20, 40, 40]]), np.array([0.8]))]
    assert M.detection_ap(loose, gt, 0.5) == 1.0 and M.detection_ap(loose, gt, 0.8) == pytest.approx(0.25)
    assert 0 < M.detection_map50_95(loose, gt) < 1


def test_type_report_uses_mask_and_within_waste():
    yw = np.array([1, 1, 1, 1, 0, 0]); sw = np.array([.9, .8, .7, .6, .2, .1])
    yt = np.array([[1], [0], [1], [0], [0], [0]], float)
    st = np.array([[.9], [.1], [.2], [.95], [.05], [.05]])
    mask = np.array([[1], [1], [1], [0], [1], [1]], float)   # 4th image: type unknown -> ignored
    rep = M.classification_report(yw, sw, yt, st, mask, ["Plastic"], 0.5)
    t = rep["types"]["Plastic"]
    assert t["images"] == 5 and t["positives"] == 2
    assert t["ap"] == pytest.approx(M.average_precision([1, 0, 1, 0, 0], [.9, .1, .2, .05, .05]))
    assert t["ap_within_waste"] == pytest.approx(M.average_precision([1, 0, 1], [.9, .1, .2]))


def test_masked_types_do_not_train():
    logits = torch.zeros(2, 3, requires_grad=True)
    yw = torch.tensor([1.0, 0.0]); yt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    loss, _ = multitask_loss(logits, yw, yt, mask)
    loss.backward()
    g = logits.grad
    assert g[0, 1] != 0 and g[0, 2] == 0 and g[1, 1] == 0 and g[1, 2] == 0
    assert g[1, 0] != 0   # the waste output always trains


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); s = y * 0.3 + rng.random(300)
    lo, hi = M.bootstrap_ci(y, s, n=300)
    assert lo < M.average_precision(y, s) < hi and hi - lo < 0.3


def test_white_column_index():
    from wastedet.baselines import FEAT_SIZE, WHITE_COL, handcrafted_features, white_fraction
    from PIL import Image

    img = (np.random.default_rng(1).random((300, 300, 3)) * 255).astype(np.uint8)
    img[:120, :150] = 250
    small = np.asarray(Image.fromarray(img).resize((FEAT_SIZE, FEAT_SIZE), Image.BILINEAR))
    assert handcrafted_features(img)[WHITE_COL] == pytest.approx(white_fraction(small))
