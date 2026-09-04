#!/usr/bin/env python3
"""評估訓練好的角點模型,協定與 eval_baseline.py 相同,可直接對比。

- 只在 GT 可見角點上算像素誤差 (原始解析度座標系)
- 額外報告 presence 準確率 (特寫鏡頭判斷,baseline 沒有這能力)
- 疊圖輸出 data/viz_model/: 綠 = 模型預測,白 = GT

用法:
    python scripts/eval_model.py [--ckpt checkpoints/best.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import FRAMES_DIR, INPUT_H, INPUT_W, TableDataset  # noqa: E402
from src.model import TableKeypointNet, decode_heatmaps  # noqa: E402

VIZ_DIR = PROJECT_ROOT / "data" / "viz_model"
ORIG_W, ORIG_H = 1280, 720
SCORE_THRESH = 0.3  # heatmap 峰值低於此視為角點不可偵測


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"載入 {args.ckpt} (epoch {ckpt['epoch']})")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    sx, sy = ORIG_W / INPUT_W, ORIG_H / INPUT_H

    for split in ("train", "val"):
        ds = TableDataset(split, augment=False)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        errors = []
        presence_correct = 0
        n_pos = n_neg = 0

        for batch in loader:
            heatmaps, presence_logit, _ = model(batch["image"].to(device))
            coords, scores = decode_heatmaps(heatmaps)
            coords = coords.cpu().numpy() * [sx, sy]
            scores = scores.cpu().numpy()
            presence_prob = torch.sigmoid(presence_logit).cpu().numpy()

            gt = batch["corners"].numpy() * [sx, sy]
            vis = batch["vis_gt"].numpy() > 0.5
            presence_gt = batch["presence"].numpy() > 0.5

            for b in range(len(presence_gt)):
                pred_present = presence_prob[b] > 0.5
                presence_correct += int(pred_present == presence_gt[b])
                if presence_gt[b]:
                    n_pos += 1
                    err = np.linalg.norm(coords[b] - gt[b], axis=1)
                    errors.extend(err[vis[b]])
                else:
                    n_neg += 1

                # 疊圖 (val 全存,train 只存有大誤差的)
                if split == "val" or (presence_gt[b] and vis[b].any()
                                      and np.linalg.norm(coords[b] - gt[b], axis=1)[vis[b]].max() > 20):
                    img = cv2.imread(str(FRAMES_DIR / f"{batch['stem'][b]}.jpg"))
                    if presence_gt[b]:
                        cv2.polylines(img, [gt[b].astype(np.int32)], True, (255, 255, 255), 2)
                    if pred_present:
                        for i in range(4):
                            color = (0, 255, 0) if scores[b, i] >= SCORE_THRESH else (0, 0, 255)
                            cv2.circle(img, tuple(coords[b, i].astype(int)), 8, color, -1)
                        cv2.polylines(img, [coords[b].astype(np.int32)], True, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, f"NO TABLE ({presence_prob[b]:.2f})", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                    cv2.imwrite(str(VIZ_DIR / f"{batch['stem'][b]}.jpg"), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

        errors = np.array(errors)
        print(f"\n[{split}]  正樣本 {n_pos} / 負樣本 {n_neg}")
        print(f"  presence 準確率: {presence_correct / (n_pos + n_neg):.0%}")
        if len(errors):
            print(f"  可見角點誤差 (px): mean={errors.mean():.1f}  median={np.median(errors):.1f}  "
                  f"p95={np.percentile(errors, 95):.1f}  max={errors.max():.1f}")
            print(f"  <5px: {(errors < 5).mean():.0%}   <10px: {(errors < 10).mean():.0%}")

    print(f"\n疊圖輸出於 {VIZ_DIR}")


if __name__ == "__main__":
    main()
