#!/usr/bin/env python3
"""輸出帶「落點小視窗」的示範影片。

畫面內容:
  - 主畫面: 原始影片 + 本專案偵測的桌面四邊形 (逐幀)
  - 右下小視窗: 俯視桌面圖,累積顯示已偵測到的落點;
    偵測到新落點時該點會放大閃爍並顯示分區 (同時在主畫面標出位置)

落點事件讀自 scripts/landing_points.py 產生的 JSON (不重跑球偵測),
桌面四邊形則重跑 TableTracker (每幀約數毫秒)。

用法:
    python scripts/render_landing_demo.py data/landing_xxx.json
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

from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

TABLE_W, TABLE_H = 500.0, 240.0   # 同教授的桌面座標空間
FLASH_SEC = 1.2                    # 新落點閃爍持續秒數
MAP_SCALE = 0.62                   # 小視窗中桌面的縮放
PAD = 16                           # 小視窗內邊距


def draw_minimap(canvas, bounces, now, origin):
    """在 canvas 右下畫俯視桌面小視窗。bounces: 已發生的落點清單。"""
    tw, th = int(TABLE_W * MAP_SCALE), int(TABLE_H * MAP_SCALE)
    pw, ph = tw + PAD * 2, th + PAD * 2 + 22
    ox, oy = origin

    # 半透明底板
    panel = canvas[oy:oy + ph, ox:ox + pw]
    panel[:] = (panel * 0.25 + np.array([25, 25, 25]) * 0.75).astype(np.uint8)
    cv2.rectangle(canvas, (ox, oy), (ox + pw, oy + ph), (200, 200, 200), 1)

    tx0, ty0 = ox + PAD, oy + PAD + 22
    cv2.rectangle(canvas, (tx0, ty0), (tx0 + tw, ty0 + th), (60, 95, 60), -1)
    for i in range(1, 3):  # 3x3 分區格線
        y = ty0 + int(th * i / 3)
        cv2.line(canvas, (tx0, y), (tx0 + tw, y), (110, 110, 110), 1)
    for half in (0, 1):
        for i in range(1, 3):
            x = tx0 + int(tw * (half * 3 + i) / 6)
            cv2.line(canvas, (x, ty0), (x, ty0 + th), (110, 110, 110), 1)
    cv2.rectangle(canvas, (tx0, ty0), (tx0 + tw, ty0 + th), (255, 255, 255), 1)
    cv2.line(canvas, (tx0 + tw // 2, ty0), (tx0 + tw // 2, ty0 + th), (0, 200, 255), 2)
    cv2.putText(canvas, "FAR", (tx0 + 3, ty0 + 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NEAR", (tx0 + 3, ty0 + th - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (180, 180, 180), 1, cv2.LINE_AA)

    label = f"BOUNCES  {len(bounces)}"
    flash_zone = None
    for e in bounces:
        # 垂直翻轉: 桌面座標 y=0 是近端,但影片中近端在畫面下方,翻轉後小視窗與影片同向
        px = tx0 + int(e["table"][0] * MAP_SCALE)
        py = ty0 + th - int(e["table"][1] * MAP_SCALE)
        age = now - e["t"]
        if 0 <= age < FLASH_SEC:  # 新落點: 放大 + 擴散環
            r = int(6 + 14 * (age / FLASH_SEC))
            cv2.circle(canvas, (px, py), r, (0, 255, 255), 2)
            cv2.circle(canvas, (px, py), 6, (0, 255, 255), -1)
            flash_zone = e.get("zone")
        else:                      # 歷史落點: 暗紅小點
            cv2.circle(canvas, (px, py), 4, (90, 90, 230), -1)
            cv2.circle(canvas, (px, py), 4, (220, 220, 220), 1)

    if flash_zone:
        label = f"BOUNCE  zone {flash_zone}"
    color = (0, 255, 255) if flash_zone else (220, 220, 220)
    cv2.putText(canvas, label, (tx0, oy + PAD + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text())
    bounces = [e for e in data["events"]
               if e.get("type") == "bounce" and e.get("table")]
    video, fps = data["video"], data["fps"]
    start, dur = data["start"], data["dur"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tracker = TableTracker(w, h)

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"demo_{Path(video).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    map_w = int(TABLE_W * MAP_SCALE) + PAD * 2
    map_h = int(TABLE_H * MAP_SCALE) + PAD * 2 + 22
    origin = (w - map_w - 20, h - map_h - 20)   # 右下角

    for i in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        now = (start_f + i) / fps

        coords, scores, presence = predict(model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        if quad is not None:
            cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 255, 0), 2)

        # 主畫面: 閃爍中的落點位置
        for e in bounces:
            age = now - e["t"]
            if 0 <= age < FLASH_SEC:
                x, y = int(e["x"]), int(e["y"])
                r = int(14 + 26 * (age / FLASH_SEC))
                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)

        draw_minimap(frame, [e for e in bounces if e["t"] <= now], now, origin)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"示範影片輸出於 {out_path}  (落點 {len(bounces)} 個)")


if __name__ == "__main__":
    main()
