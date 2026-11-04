"""Models.

1. WasteClassifier - looks at one 210 m x 210 m window and answers
   "is there a waste dump here?" plus "which waste types are visible?" (Plastic, Rubble, ...).
   Backbone from timm (Apache-2.0), ImageNet-pretrained ConvNeXt by default.
2. Dumpsite detector - torchvision Faster R-CNN (BSD licence) that draws boxes around dumps.
   Chosen instead of Ultralytics YOLO, whose AGPL-3.0 licence is a problem for company code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalised-mean pooling: between average and max pooling. Helps when the waste
    occupies only a small part of the window (average pooling would dilute it)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class WasteClassifier(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny.fb_in22k_ft_in1k", n_types: int = 7,
                 pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        dim = self.backbone.num_features
        self.pool = GeM()
        self.drop = nn.Dropout(drop)
        self.head_waste = nn.Linear(dim, 1)
        self.head_types = nn.Linear(dim, n_types)
        self.n_types = n_types
        self.backbone_name = backbone

    def features(self, x):
        return self.backbone.forward_features(x)  # (B, C, h, w)

    def forward(self, x):
        f = self.drop(self.pool(self.features(x)))
        # one output tensor (logits): column 0 = waste, columns 1.. = types (ONNX-friendly)
        return torch.cat([self.head_waste(f), self.head_types(f)], dim=1)

    @torch.no_grad()
    def cam(self, x):
        """Class-activation map for the 'waste' output: where in the window the model looks."""
        f = self.features(x)
        w = self.head_waste.weight[0].view(1, -1, 1, 1)
        cam = F.relu((f * w).sum(1))
        cam = cam / cam.flatten(1).amax(1).clamp(min=1e-6).view(-1, 1, 1)
        return F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


def multitask_loss(logits, y_waste, y_types, type_mask, pos_weight: float = 1.0,
                   type_pos_weight: torch.Tensor | None = None, type_weight: float = 0.5):
    """BCE for 'waste' on every image; BCE for types only where the types are known."""
    lw = F.binary_cross_entropy_with_logits(
        logits[:, 0], y_waste, pos_weight=torch.tensor(pos_weight, device=logits.device))
    lt_all = F.binary_cross_entropy_with_logits(
        logits[:, 1:], y_types, reduction="none",
        pos_weight=None if type_pos_weight is None else type_pos_weight.to(logits.device))
    m = type_mask.view(-1, 1).expand_as(lt_all) if type_mask.dim() == 1 else type_mask
    lt = (lt_all * m).sum() / m.sum().clamp(min=1.0)
    return lw + type_weight * lt, {"waste": float(lw.detach()), "types": float(lt.detach())}


def build_detector(pretrained: bool = True, min_size: int = 800, max_size: int = 800):
    """Faster R-CNN ResNet-50 FPN v2, COCO-pretrained, new 2-class head (background, dumpsite)."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = "DEFAULT" if pretrained else None
    weights_backbone = None
    m = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=weights_backbone,
                                   min_size=min_size, max_size=max_size, box_detections_per_img=50)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    return m


def export_classifier_onnx(model: WasteClassifier, path, size: int = 640, check: bool = True) -> float:
    """Export to ONNX (opset 17, dynamic batch) and check it matches PyTorch.
    Returns the max relative difference."""
    import numpy as np

    model = model.eval().cpu()
    dummy = torch.randn(2, 3, size, size)
    torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    if not check:
        return float("nan")
    import onnxruntime as ort

    out = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": dummy.numpy()})[0]
    ref = model(dummy).detach().numpy()
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-8))
    assert rel < 1e-3, f"ONNX output differs from PyTorch (max rel diff {rel:.2e})"
    return rel
