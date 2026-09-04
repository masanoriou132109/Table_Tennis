#!/usr/bin/env python3
"""訓練桌面 4 角點模型。

損失 = heatmap MSE (x100) + presence BCE + visibility BCE (僅正樣本)
每個 epoch 在 val 上解碼角點算像素誤差 (原始 1280x720 座標系,只算 GT 可見角),
以 val median 誤差存最佳權重到 checkpoints/best.pt。

用法:
    python scripts/train.py [--epochs 150] [--batch 8] [--lr 1e-3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import INPUT_H, INPUT_W, TableDataset  # noqa: E402
from src.model import TableKeypointNet, decode_heatmaps, heatmap_focal_loss  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "checkpoints"
ORIG_W, ORIG_H = 1280, 720


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_loss(model_out, batch, device):
    heatmaps, presence_logit, vis_logits = model_out
    hm_target = batch["heatmaps"].to(device)
    presence = batch["presence"].to(device)
    vis = batch["visibility"].to(device)

    loss_hm = heatmap_focal_loss(heatmaps, hm_target)
    loss_presence = F.binary_cross_entropy_with_logits(presence_logit, presence)
    pos = presence > 0.5
    loss_vis = (F.binary_cross_entropy_with_logits(vis_logits[pos], vis[pos])
                if pos.any() else torch.zeros((), device=device))
    return loss_hm + loss_presence + loss_vis


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errors, presence_correct, total = [], 0, 0
    for batch in loader:
        out = model(batch["image"].to(device))
        heatmaps, presence_logit, _ = out
        coords, _ = decode_heatmaps(heatmaps)
        coords = coords.cpu().numpy() * [ORIG_W / INPUT_W, ORIG_H / INPUT_H]

        gt = batch["corners"].numpy() * [ORIG_W / INPUT_W, ORIG_H / INPUT_H]
        vis = batch["vis_gt"].numpy() > 0.5
        presence_gt = batch["presence"].numpy() > 0.5
        presence_pred = (torch.sigmoid(presence_logit).cpu().numpy() > 0.5)

        presence_correct += int((presence_pred == presence_gt).sum())
        total += len(presence_gt)
        for b in range(len(presence_gt)):
            if presence_gt[b]:
                err = np.linalg.norm(coords[b] - gt[b], axis=1)
                errors.extend(err[vis[b]])
    errors = np.array(errors)
    return {
        "median": float(np.median(errors)) if len(errors) else float("inf"),
        "mean": float(errors.mean()) if len(errors) else float("inf"),
        "lt10": float((errors < 10).mean()) if len(errors) else 0.0,
        "presence_acc": presence_correct / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    train_ds = TableDataset("train", augment=True)
    val_ds = TableDataset("val", augment=False)
    print(f"train {len(train_ds)} 張 / val {len(val_ds)} 張 (含負樣本)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = TableKeypointNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CKPT_DIR.mkdir(exist_ok=True)
    best_median = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model(batch["image"].to(device)), batch, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            m = evaluate(model, val_loader, device)
            mark = ""
            if m["median"] < best_median:
                best_median = m["median"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": m},
                           CKPT_DIR / "best.pt")
                mark = "  <- best"
            print(f"epoch {epoch:3d}  loss {epoch_loss / len(train_loader):.4f}  "
                  f"val median {m['median']:5.1f}px  <10px {m['lt10']:4.0%}  "
                  f"presence {m['presence_acc']:4.0%}  ({time.time() - t0:.0f}s){mark}")

    print(f"\n完成。最佳 val median = {best_median:.1f}px,權重在 {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
