#!/usr/bin/env python3
"""持續遮擋測試: 量化遮擋角重建方法隨遮擋時長的誤差累積。

找出「連續 W+1 格四角皆高信心」的視窗當 pseudo-GT。
模擬某角在整個視窗被遮,逐幀迭代重建 (如真實平滑器: 可見角追隨真值,
被擋角用前一重建狀態 + 本格可見角的平面變換推算),
比較重建位置 vs 模型真實高信心預測,依「遮擋幀數」統計誤差。

比較: hold(沿用) / similarity(2點相似) / affine(3點完整仿射)。

用法:
    python scripts/eval_occlusion_fill.py "videos/xxx.mp4" [--start 0 --dur 300 --window 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TableKeypointNet  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

MODES = ("hold", "similarity", "affine", "hybrid")
MOTION_REF = 3.0  # px: 可見角平均位移達此值即完全採用 affine


def apply_transform(prev4, cur_visible_true, drop, mode):
    """prev4: 上一重建狀態(4角); cur_visible_true: 本格可見角真值。"""
    keep = [i for i in range(4) if i != drop]
    src = prev4[keep].astype(np.float32)
    dst = cur_visible_true[keep].astype(np.float32)
    hold = prev4[drop].copy()
    if mode == "hold":
        return hold
    if mode == "similarity":
        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    else:  # affine / hybrid 皆用完整仿射
        M = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)[0]
    if M is None:
        return hold
    warped = M @ np.array([prev4[drop][0], prev4[drop][1], 1.0])
    if mode == "hybrid":
        motion = float(np.linalg.norm(dst - src, axis=1).mean())
        w = min(motion / MOTION_REF, 1.0)  # 靜止→hold, 運鏡→affine
        return (1 - w) * hold + w * warped
    return warped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--dur", type=float, default=300)
    ap.add_argument("--window", type=int, default=30, help="模擬遮擋持續幀數")
    ap.add_argument("--score-thresh", type=float, default=0.35)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(PROJECT_ROOT / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))

    # 先蒐集整段每格「四角皆可信」的角座標序列 (斷開處以 None 分段)
    seq: list[np.ndarray | None] = []
    for _ in range(int(args.dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        coords, scores, presence = predict(model, frame, device, w, h)
        seq.append(coords.copy() if (presence >= 0.5 and (scores >= args.score_thresh).all()) else None)
    cap.release()

    W = args.window
    # 依遮擋幀數 j 累積誤差
    err = {m: [[] for _ in range(W)] for m in MODES}
    for s in range(len(seq) - W):
        block = seq[s:s + W + 1]
        if any(b is None for b in block):  # 需整段皆可信
            continue
        block = [b for b in block]
        for drop in range(4):
            for mode in MODES:
                state = block[0].copy()  # t=s 全真
                for j in range(1, W + 1):
                    cur_true = block[j]
                    new = cur_true.copy()          # 3 可見角追隨真值
                    new[drop] = apply_transform(state, cur_true, drop, mode)
                    err[mode][j - 1].append(float(np.linalg.norm(new[drop] - cur_true[drop])))
                    state = new
    n = len(err["hold"][0]) if err["hold"][0] else 0
    print(f"視窗數: {n}  (每視窗 4 角 × {W} 幀)")
    print(f"{'遮擋幀':>6s} " + " ".join(f"{m:>10s}" for m in MODES))
    for j in (1, 5, 10, 15, 20, W):
        if j <= W:
            row = " ".join(f"{np.mean(err[m][j-1]):10.1f}" for m in MODES)
            print(f"{j:6d} {row}  (px, mean)")


if __name__ == "__main__":
    main()
