#!/usr/bin/env python3
"""訓練桌面 4 角點模型。

損失 = heatmap MSE (x100) + presence BCE + visibility BCE (僅正樣本)
每個 epoch 在 val 上解碼角點算像素誤差 (原始 1280x720 座標系,只算 GT 可見角),
以 val 誤差存最佳權重到 --out (預設 checkpoints/best.pt)。

v3: 被擋的角也有 heatmap 目標, 評估分開報「看得到的角」與「被擋的角」;
最佳權重以「全部角 (含被擋) 的平均誤差」挑 (平均對誤認角這種大誤差敏感, 正是要壓的)。
--init 從既有權重 fine-tune; --v3-frac 只用部分 v3 訓練資料 (學習曲線用)。

用法:
    python scripts/train.py [--epochs 150] [--batch 8] [--lr 1e-3]
    python scripts/train.py --init checkpoints/v2_best.pt --out checkpoints/v3_best.pt --epochs 150 --lr 3e-4
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
    errors, occ_errors, groups, presence_correct, total = [], [], {}, 0, 0
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
        in_frame = ((gt[..., 0] >= 0) & (gt[..., 0] < ORIG_W) & (gt[..., 1] >= 0) & (gt[..., 1] < ORIG_H))
        for b in range(len(presence_gt)):
            if presence_gt[b]:
                err = np.linalg.norm(coords[b] - gt[b], axis=1)
                errors.extend(err[vis[b]])
                occ_errors.extend(err[~vis[b] & in_frame[b]])
                g = groups.setdefault("v3" if batch["stem"][b].startswith("v3_") else "v2", [])
                g.extend(err[in_frame[b]])
    errors, occ_errors = np.array(errors), np.array(occ_errors)
    every = np.concatenate([errors, occ_errors])
    return {
        "median": float(np.median(errors)) if len(errors) else float("inf"),
        "mean": float(errors.mean()) if len(errors) else float("inf"),
        "lt10": float((errors < 10).mean()) if len(errors) else 0.0,
        "occ_median": float(np.median(occ_errors)) if len(occ_errors) else float("nan"),
        "occ_n": len(occ_errors),
        "all_mean": float(every.mean()) if len(every) else float("inf"),
        "groups": {k: (float(np.median(v)), float(np.mean(v)), float((np.array(v) > 20).mean()), len(v))
                   for k, v in sorted(groups.items())},
        "presence_acc": presence_correct / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", help="從這個權重 fine-tune (預設: ImageNet 初始化)")
    parser.add_argument("--out", default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--v3-frac", type=float, default=1.0, help="只用這個比例的 v3 訓練樣本")
    parser.add_argument("--occlusion", type=float, default=0.5, help="合成遮擋機率")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out_path = Path(args.out)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    train_ds = TableDataset("train", augment=True, occlusion=args.occlusion)
    if args.v3_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        v3 = [i for i, s in enumerate(train_ds.samples) if s["path"].stem.startswith("v3_")]
        drop = set(rng.choice(v3, size=len(v3) - int(round(len(v3) * args.v3_frac)), replace=False).tolist())
        train_ds.samples = [s for i, s in enumerate(train_ds.samples) if i not in drop]
    val_ds = TableDataset("val", augment=False)
    print(f"train {len(train_ds)} 張 / val {len(val_ds)} 張 (含負樣本)")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = TableKeypointNet(pretrained=args.init is None).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        m = evaluate(model, val_loader, device)
        print(f"初始權重 {args.init}: val 可見 median {m['median']:.1f}px  被擋 median {m['occ_median']:.1f}px "
              f"(n={m['occ_n']})  全部 mean {m['all_mean']:.1f}px")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CKPT_DIR.mkdir(exist_ok=True)
    best_score, best = float("inf"), None
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
            if m["all_mean"] < best_score:
                best_score, best = m["all_mean"], dict(m, epoch=epoch)
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": m}, out_path)
                mark = "  <- best"
            print(f"epoch {epoch:3d}  loss {epoch_loss / len(train_loader):.4f}  "
                  f"val 可見 median {m['median']:5.1f}px  <10px {m['lt10']:4.0%}  "
                  f"被擋 median {m['occ_median']:5.1f}px  全部 mean {m['all_mean']:5.1f}px  "
                  f"presence {m['presence_acc']:4.0%}  ({time.time() - t0:.0f}s){mark}", flush=True)

    print(f"\n完成。最佳 epoch {best['epoch']}: 全部 mean {best_score:.1f}px, 權重在 {out_path}")
    for k, (med, mean, gt20, n) in best["groups"].items():
        print(f"  {k}: median {med:.1f}px  mean {mean:.1f}px  >20px {gt20:.1%}  (n={n} 角)")


if __name__ == "__main__":
    main()
