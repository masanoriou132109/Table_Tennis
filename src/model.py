"""輕量桌面 4 角點偵測模型 (SimpleBaseline 風格)。

ResNet18 backbone (stride 32) -> 3 層反卷積 (stride 4) -> 4 通道角點 heatmap
另從 backbone 特徵 global pool 出兩個小 head:
  - presence: 畫面中是否有桌面 (1 logit) — 處理特寫鏡頭
  - visibility: 各角是否可見且在畫面內 (4 logits)

輸入 512x288 (1280x720 等比縮 0.4),heatmap 輸出 128x72。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

NUM_CORNERS = 4
HEATMAP_STRIDE = 4  # 相對輸入解析度


class TableKeypointNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 512, H/32, W/32]

        deconv_layers = []
        in_ch = 512
        for out_ch in (256, 256, 256):  # stride 32 -> 4
            deconv_layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.deconv = nn.Sequential(*deconv_layers)
        self.heatmap_head = nn.Conv2d(256, NUM_CORNERS, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1 + NUM_CORNERS),  # presence + visibility
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        heatmaps = self.heatmap_head(self.deconv(feat))
        logits = self.cls_head(self.pool(feat))
        presence_logit = logits[:, 0]
        visibility_logits = logits[:, 1:]
        return heatmaps, presence_logit, visibility_logits


def decode_heatmaps(heatmaps: torch.Tensor, window: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """從 heatmap 取各角座標 (輸入影像座標系) 與峰值分數。
    先 argmax 找峰,再取 (2*window+1)^2 鄰域的機率加權質心做 subpixel 精修,
    消除 stride-4 的量化誤差。
    heatmaps: [B, 4, Hh, Wh] (未過 sigmoid)。回傳 coords [B,4,2], scores [B,4]。
    """
    b, c, hh, wh = heatmaps.shape
    probs = torch.sigmoid(heatmaps)
    flat = probs.reshape(b, c, -1)
    scores, idx = flat.max(dim=2)
    ys = (idx // wh).float()
    xs = (idx % wh).float()

    coords = torch.empty(b, c, 2, device=heatmaps.device)
    for bi in range(b):
        for ci in range(c):
            px, py = int(xs[bi, ci]), int(ys[bi, ci])
            x0, x1 = max(px - window, 0), min(px + window + 1, wh)
            y0, y1 = max(py - window, 0), min(py + window + 1, hh)
            patch = probs[bi, ci, y0:y1, x0:x1]
            total = patch.sum()
            gx = torch.arange(x0, x1, device=patch.device, dtype=torch.float32)
            gy = torch.arange(y0, y1, device=patch.device, dtype=torch.float32)
            coords[bi, ci, 0] = (patch.sum(0) * gx).sum() / total
            coords[bi, ci, 1] = (patch.sum(1) * gy).sum() / total
    coords = coords * HEATMAP_STRIDE + HEATMAP_STRIDE / 2
    return coords, scores


def heatmap_focal_loss(pred_logits: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet 式 penalty-reduced focal loss (target 為高斯 heatmap)。
    比 MSE 更能抑制假峰值、銳化真峰值,小資料下差異顯著。"""
    p = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    pos = target > 0.999
    pos_loss = -((1 - p) ** alpha * torch.log(p))[pos].sum()
    neg_loss = -((1 - target) ** beta * p ** alpha * torch.log(1 - p))[~pos].sum()
    num_pos = pos.sum().clamp(min=1)
    return (pos_loss + neg_loss) / num_pos
