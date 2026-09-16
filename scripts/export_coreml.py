#!/usr/bin/env python3
"""把訓練好的 TableKeypointNet 轉成 Core ML (供 iOS/macOS 即時推論)。

為了讓 Swift 端盡量單純,把前後處理都包進模型:
  - 輸入: 512x288 RGB 影像 (Core ML ImageType, 0~255) → 模型內部做 /255 與 ImageNet 正規化
  - 輸出: 已過 sigmoid 的機率
      heatmaps    [1,4,72,128]  4 個角的 heatmap (far_left, far_right, near_right, near_left)
      presence    [1]           畫面是否有桌面
      visibility  [1,4]         各角可見性 (輔助, 實務上以 heatmap 峰值為準)

Swift 端只需: argmax 取峰 → 鄰域加權質心做 subpixel → 乘回原圖尺度,
再套 TableTracker 的平滑/先驗補角/閘門邏輯 (src/tracker.py 的 Swift 移植)。

用法:
    python scripts/export_coreml.py [--ckpt checkpoints/best.pt] [--out Models/TableDetector.mlpackage]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, INPUT_H, INPUT_W  # noqa: E402
from src.model import TableKeypointNet  # noqa: E402


class DeployWrapper(nn.Module):
    """包正規化 + sigmoid,讓 Core ML 端輸入原始像素、輸出機率。"""

    def __init__(self, core: TableKeypointNet):
        super().__init__()
        self.core = core
        # Core ML ImageType 餵進來是 0~255 的 RGB
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255.0)

    def forward(self, image: torch.Tensor):
        x = (image - self.mean) / self.std
        heatmaps, presence_logit, visibility_logits = self.core(x)
        return (torch.sigmoid(heatmaps),
                torch.sigmoid(presence_logit),
                torch.sigmoid(visibility_logits))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "Models" / "TableDetector.mlpackage"))
    ap.add_argument("--deployment", default="iOS16", choices=("iOS15", "iOS16", "iOS17"))
    args = ap.parse_args()

    import coremltools as ct

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    core = TableKeypointNet(pretrained=False)
    core.load_state_dict(ckpt["model"])
    model = DeployWrapper(core).eval()

    example = torch.randint(0, 255, (1, 3, INPUT_H, INPUT_W), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        ref = model(example)

    target = {"iOS15": ct.target.iOS15, "iOS16": ct.target.iOS16,
              "iOS17": ct.target.iOS17}[args.deployment]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, INPUT_H, INPUT_W),
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="heatmaps"),
                 ct.TensorType(name="presence"),
                 ct.TensorType(name="visibility")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
    )

    mlmodel.short_description = (
        "桌球桌面 4 角點偵測 (far_left, far_right, near_right, near_left)。"
        "輸出已過 sigmoid: heatmaps[1,4,72,128]、presence[1]、visibility[1,4]。")
    mlmodel.author = "Tabletennis table detection (v2, 216 張/6 場地, val median 2.0px)"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    print(f"已輸出 {args.out}")

    # 數值一致性檢查: 同一張輸入比對 PyTorch vs Core ML
    from PIL import Image
    arr = example[0].permute(1, 2, 0).numpy().astype(np.uint8)
    pred = mlmodel.predict({"image": Image.fromarray(arr)})
    for name, t in zip(("heatmaps", "presence", "visibility"), ref):
        a = np.asarray(pred[name], np.float64).reshape(-1)
        b = t.detach().numpy().astype(np.float64).reshape(-1)
        print(f"  {name:10s} 最大絕對誤差 {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    main()
