#!/usr/bin/env python3
"""把 landing_points.py 的結果畫成俯視桌面落點圖 (含 3x3 分區與統計)。

用法:
    python scripts/plot_landing_map.py data/landing_xxx.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

TABLE_W, TABLE_H = 500.0, 240.0   # 同教授 detect_events.py 的桌面座標空間
SCALE = 2
MARGIN = 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text())
    bounces = [e for e in data["events"]
               if e.get("type") == "bounce" and e.get("table")]

    W, H = int(TABLE_W * SCALE), int(TABLE_H * SCALE)
    img = np.full((H + 2 * MARGIN, W + 2 * MARGIN, 3), 30, np.uint8)
    ox, oy = MARGIN, MARGIN

    # 桌面
    cv2.rectangle(img, (ox, oy), (ox + W, oy + H), (60, 90, 60), -1)
    cv2.rectangle(img, (ox, oy), (ox + W, oy + H), (255, 255, 255), 2)
    # 3x3 分區格線 (每半邊)
    for i in range(1, 3):
        y = oy + int(H * i / 3)
        cv2.line(img, (ox, y), (ox + W, y), (120, 120, 120), 1)
    for half in (0, 1):
        for i in range(1, 3):
            x = ox + int(W * (half * 3 + i) / 6)
            cv2.line(img, (x, oy), (x, oy + H), (120, 120, 120), 1)
    cv2.line(img, (ox + W // 2, oy), (ox + W // 2, oy + H), (0, 200, 255), 3)  # 球網

    # 落點 (依時間漸層)
    for i, e in enumerate(bounces):
        tx, ty = e["table"]
        px, py = ox + int(tx * SCALE), oy + int(ty * SCALE)
        frac = i / max(len(bounces) - 1, 1)
        col = (int(255 * (1 - frac)), 80, int(255 * frac))  # 藍→紅 = 早→晚
        cv2.circle(img, (px, py), 7, col, -1)
        cv2.circle(img, (px, py), 7, (255, 255, 255), 1)

    cv2.putText(img, "NEAR (y=0)", (ox, oy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "FAR (y=240)", (ox, oy + H + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, f"bounces: {len(bounces)}", (ox + W - 170, oy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    out = Path(args.out) if args.out else Path(args.json_path).with_suffix(".map.jpg")
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    zones = Counter(e.get("zone") for e in bounces)
    print(f"落點 {len(bounces)} 個")
    print("分區統計:", dict(sorted((k, v) for k, v in zones.items() if k)))
    off = zones.get(None, 0)
    if off:
        print(f"  (另有 {off} 個落在桌面外,未計入分區)")
    print(f"輸出 {out}")


if __name__ == "__main__":
    main()
