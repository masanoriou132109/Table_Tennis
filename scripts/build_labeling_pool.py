#!/usr/bin/env python3
"""從新影片建立「模型預標」的標註池,供 labelme 修正 (而非從零標註)。

流程:
  1. 對指定影片依 --interval 秒抽幀
  2. 用 v1 模型的 presence head 只留下有桌面的影格 (濾掉特寫/重播)
  3. 縮圖差異去重複,確保多樣性
  4. 每部影片最多留 --cap 幀
  5. 影格存到 data/frames_pool/,並用 v1 角點預測寫成 labelme JSON 到 data/labelme_pool/

之後在 labelme 開 data/frames_pool、output 指到 data/labelme_pool,
只需「修正」預標的四邊形 (Vegas/WTT 幾乎不用改,Doha 需多修),再跑 convert_labelme.py。

用法:
    python scripts/build_labeling_pool.py "videos/xxx.mp4" [更多影片...] \
        --interval 2.0 --cap 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_frames import frame_signature, short_stem  # noqa: E402
from scripts.infer_video import predict  # noqa: E402
from src.baseline_cv import detect_table_quad  # noqa: E402
from src.dataset import CORNER_NAMES  # noqa: E402
from src.model import TableKeypointNet  # noqa: E402

FRAMES_POOL = PROJECT_ROOT / "data" / "frames_pool"
LABELME_POOL = PROJECT_ROOT / "data" / "labelme_pool"


def to_labelme(frame_name: str, coords: np.ndarray, w: int, h: int) -> dict:
    """把 v1 角點預測寫成 labelme polygon (label=table, 4 點, 模型順序)。"""
    return {
        "version": "5.0.0",
        "flags": {name: False for name in CORNER_NAMES},  # 修正時勾選=遮擋
        "shapes": [{
            "label": "table",
            "points": [[round(float(x), 1), round(float(y), 1)] for x, y in coords],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }],
        "imagePath": f"../frames_pool/{frame_name}",  # JSON 在 labelme_pool,圖在 frames_pool
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--interval", type=float, default=2.0, help="抽幀間隔秒")
    parser.add_argument("--cap", type=int, default=40, help="每部影片最多留幾幀")
    parser.add_argument("--presence-thresh", type=float, default=0.6)
    parser.add_argument("--diff-thresh", type=float, default=6.0)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    FRAMES_POOL.mkdir(parents=True, exist_ok=True)
    LABELME_POOL.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for video in args.videos:
        vpath = Path(video)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[跳過] 無法開啟 {vpath.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(fps * args.interval))
        stem = short_stem(vpath)

        kept = 0
        prev_sig = None
        for fi in range(0, total, step):
            if kept >= args.cap:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            coords, scores, presence = predict(model, frame, device, w, h)
            if presence < args.presence_thresh:
                continue
            # 預標器: magenta 桌用 CV baseline (median~2px),失敗才退回 v1 角點
            quad = detect_table_quad(frame)
            if quad is not None:
                coords = quad
            sig = frame_signature(frame)
            if prev_sig is not None and float(np.abs(sig - prev_sig).mean()) < args.diff_thresh:
                continue
            prev_sig = sig

            name = f"{stem}_f{fi:06d}.jpg"
            cv2.imwrite(str(FRAMES_POOL / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (LABELME_POOL / f"{stem}_f{fi:06d}.json").write_text(
                json.dumps(to_labelme(name, coords, w, h), ensure_ascii=False, indent=2))
            kept += 1
        cap.release()
        grand_total += kept
        print(f"  {vpath.name[:45]}...  -> {kept} 幀")

    print(f"\n完成: 共 {grand_total} 幀 (含 v1 預標) 於 {FRAMES_POOL}")
    print(f"下一步: labelme \"{FRAMES_POOL}\" --output \"{LABELME_POOL}\" "
          f"--flags {','.join(CORNER_NAMES)} --nodata --autosave")


if __name__ == "__main__":
    main()
