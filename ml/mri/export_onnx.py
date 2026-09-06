"""Export the trained MRI classifier to ONNX with a CAM output.

The deployed head is global-avg-pool -> linear, for which class activation
mapping is exact without gradients:

    CAM_c(h, w) = sum_k fc.weight[c, k] * layer4_features[k, h, w]

so the graph returns (logits, cam) from one forward pass and the backend can
render "where the model looked" heatmaps under plain onnxruntime.

Standalone re-export (after training already produced mri_model.pt):
    python ml/mri/export_onnx.py [--artifacts ml/artifacts]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ResNetWithCam(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.fc = resnet.fc

    def forward(self, x):
        feat = self.trunk(x)                     # B,512,7,7
        # mean over H,W == the net's global avg pool, but stays batch-agnostic
        # in the ONNX graph (flatten+reshape bakes the batch size in).
        logits = self.fc(feat.mean(dim=(2, 3)))  # B,C
        cam = torch.einsum("ck,bkhw->bchw", self.fc.weight, feat)
        return logits, cam


def export(artifacts_dir: Path) -> Path:
    ckpt = torch.load(artifacts_dir / "mri_model.pt", map_location="cpu", weights_only=False)
    n_classes = len(ckpt["classes"])
    img_size = ckpt["img_size"]

    resnet = models.resnet18()
    resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
    resnet.load_state_dict(ckpt["state_dict"])
    model = ResNetWithCam(resnet).eval()

    out = artifacts_dir / "mri_model.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, img_size, img_size),
        str(out),
        input_names=["input"],
        output_names=["logits", "cam"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out} (outputs: logits, cam)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    export(Path(ap.parse_args().artifacts))
