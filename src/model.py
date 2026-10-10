"""輕量桌面 4 角點偵測模型 (SimpleBaseline 風格)。

ResNet18 backbone (stride 32) -> 3 層反卷積 (stride 4) -> 4 通道角點 heatmap
另從 backbone 特徵 global pool 出兩個小 head:
  - presence: 畫面中是否有桌面 (1 logit) — 處理特寫鏡頭
  - visibility: 各角是否可見且在畫面內 (4 logits)

輸入 512x288 (1280x720 等比縮 0.4),heatmap 輸出 128x72。
"""

from __future__ import annotations

import numpy as np
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


def decode_prob_heatmaps(p: np.ndarray, window: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """decode_heatmaps 的 numpy 版, 輸入為「已過 sigmoid」的機率 [B,4,Hh,Wh]
    (Core ML 匯出的模型直接輸出機率)。回傳 coords [B,4,2] (輸入影像座標), scores [B,4]。
    """
    b, c, hh, wh = p.shape
    flat = p.reshape(b, c, -1)
    ix = flat.argmax(axis=2)
    scores = np.take_along_axis(flat, ix[..., None], axis=2)[..., 0]
    # 向量化計算。舊版逐角在 GPU 上取 int(), 每次都強制同步。
    # 邊界處理與舊版相同: 補 0 等同把超出邊界的鄰域截掉 (0 不貢獻權重)。
    pp = np.pad(p, ((0, 0), (0, 0), (window, window), (window, window)))
    ys, xs = ix // wh, ix % wh                            # [B, C]
    offs = np.arange(-window, window + 1)
    rows = (ys[..., None] + offs + window)[..., :, None]  # [B, C, K, 1]
    cols = (xs[..., None] + offs + window)[..., None, :]  # [B, C, 1, K]
    patch = pp[np.arange(b)[:, None, None, None], np.arange(c)[None, :, None, None],
               rows, cols]                                # [B, C, K, K]
    total = patch.sum(axis=(2, 3))
    cx = (patch.sum(axis=2) * (xs[..., None] + offs)).sum(-1) / total
    cy = (patch.sum(axis=3) * (ys[..., None] + offs)).sum(-1) / total
    coords = np.stack([cx, cy], -1) * HEATMAP_STRIDE + HEATMAP_STRIDE / 2
    return coords.astype(np.float32), scores.astype(np.float32)


def decode_heatmaps(heatmaps: torch.Tensor, window: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """從 heatmap 取各角座標 (輸入影像座標系) 與峰值分數。
    先 argmax 找峰,再取 (2*window+1)^2 鄰域的機率加權質心做 subpixel 精修,
    消除 stride-4 的量化誤差。
    heatmaps: [B, 4, Hh, Wh] (未過 sigmoid)。回傳 coords [B,4,2], scores [B,4]。
    """
    probs = torch.sigmoid(heatmaps)
    scores = probs.reshape(*probs.shape[:2], -1).max(dim=2).values
    coords, _ = decode_prob_heatmaps(probs.detach().float().cpu().numpy(), window)
    return torch.from_numpy(coords).to(heatmaps.device), scores


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
