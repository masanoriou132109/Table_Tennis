#!/usr/bin/env python3
"""輸出完整示範影片: 桌面偵測 + 球軌跡 + 落點。

畫面內容:
  - 主畫面: 原始影片 + 桌面四邊形 (逐幀偵測)
  - 球: 綠圈標出該幀進入軌跡的球 (即實際餵給落點演算法的點)
  - 左上 HUD: BALL <conf> / NO BALL + 時間戳
  - 底部時間軸: 最近 --window 秒的球軌跡覆蓋 (綠=有, 暗紅=無)
    → 可直接對照「軌跡有空隙」與「該處沒測到落點」的關係
  - 右下小視窗: 俯視桌面,累積顯示落點;新落點放大閃爍並顯示分區

資料全部讀自 scripts/landing_points.py 的 JSON (球軌跡與事件都在裡面,
不需重跑球偵測),僅重跑 TableTracker 畫桌面框 (每幀數毫秒)。

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

TABLE_W, TABLE_H = 274.0, 152.5   # ITTF 正規球桌 (cm),物理等比
FLASH_SEC = 1.2                    # 新落點閃爍持續秒數
MAP_SCALE = 1.1                    # cm → 小視窗像素
PAD = 16                           # 小視窗內邊距
STRIP_H = 20                       # 底部球軌跡時間軸高度


def draw_track_strip(frame, history, now, window_sec, x_right):
    """底部時間軸: 最近 window_sec 秒球軌跡是否有點 (綠=有, 暗紅=無)。"""
    h = frame.shape[0]
    x0, x1 = 20, x_right
    y0 = h - STRIP_H - 18
    cv2.rectangle(frame, (x0, y0), (x1, y0 + STRIP_H), (35, 35, 35), -1)
    t_start = now - window_sec
    for t, present in history:
        if t < t_start:
            continue
        px = x0 + int((t - t_start) / window_sec * (x1 - x0))
        cv2.line(frame, (px, y0 + 2), (px, y0 + STRIP_H - 2),
                 (60, 200, 60) if present else (40, 40, 90), 2)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + STRIP_H), (180, 180, 180), 1)
    cv2.putText(frame, f"ball track  -{window_sec:.0f}s", (x0, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, "now", (x1 - 32, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


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

    on_table = [e for e in bounces if e.get("zone")]
    off_table = [e for e in bounces if e.get("table") and not e.get("zone")]
    unmapped = [e for e in bounces if not e.get("table")]
    label = f"BOUNCES  {len(on_table)}"
    if off_table or unmapped:
        label += f"   (off {len(off_table)} / unmap {len(unmapped)})"

    flash = None
    for e in on_table:
        # 垂直翻轉: 桌面座標 y=0 是近端,但影片中近端在畫面下方,翻轉後小視窗與影片同向
        px = tx0 + int(e["table"][0] * MAP_SCALE)
        py = ty0 + th - int(e["table"][1] * MAP_SCALE)
        age = now - e["t"]
        if 0 <= age < FLASH_SEC:  # 新落點: 放大 + 擴散環
            r = int(6 + 14 * (age / FLASH_SEC))
            cv2.circle(canvas, (px, py), r, (0, 255, 255), 2)
            cv2.circle(canvas, (px, py), 6, (0, 255, 255), -1)
            flash = f"BOUNCE  zone {e['zone']}"
        else:                      # 歷史落點: 暗紅小點
            cv2.circle(canvas, (px, py), 4, (90, 90, 230), -1)
            cv2.circle(canvas, (px, py), 4, (220, 220, 220), 1)
    for e in off_table + unmapped:  # 可疑事件也要讓使用者看到,不靜默丟棄
        if 0 <= now - e["t"] < FLASH_SEC:
            flash = "BOUNCE  OFF-TABLE (suspect)" if e.get("table") else "BOUNCE  UNMAPPED (no table)"

    color = (0, 255, 255) if flash and "zone" in flash else \
            ((40, 170, 255) if flash else (220, 220, 220))
    cv2.putText(canvas, flash or label, (tx0, oy + PAD + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--window", type=float, default=12.0, help="時間軸顯示秒數")
    ap.add_argument("--parts", default="table,ball,strip,bounce,map",
                    help="要顯示的元件,逗號分隔: table,ball,strip,bounce,map")
    ap.add_argument("--map-pos", default="br", choices=("br", "tr", "bl", "tl"),
                    help="小視窗位置: br=右下 tr=右上 bl=左下 tl=左上")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}

    data = json.loads(Path(args.json_path).read_text())
    # 全部落點都要顯示: 桌上(正常) / 桌外(疑似擊球誤判) / 未映射(當時無桌面資訊)。
    # 先前只取有 table 的,導致未映射的落點在 demo 中完全消失。
    bounces = [e for e in data["events"] if e.get("type") == "bounce"]
    video, fps = data["video"], data["fps"]
    start, dur = data["start"], data["dur"]
    # 球軌跡: 實際餵給落點演算法的點 (JSON 內已有,不需重跑球偵測)
    track_by_frame = {p["frame"]: p for p in data.get("track", [])}

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
    M = 20
    origin = {"br": (w - map_w - M, h - map_h - M),
              "tr": (w - map_w - M, M),
              "bl": (M, h - map_h - M),
              "tl": (M, M)}[args.map_pos]

    history: list[tuple[float, bool]] = []
    for i in range(int(dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        fi = start_f + i
        now = fi / fps

        coords, scores, presence = predict(model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        if quad is not None and "table" in parts:
            cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 255, 0), 2)

        # 球 (軌跡點)
        bp = track_by_frame.get(fi)
        history.append((now, bp is not None))
        if "ball" in parts:
            if bp is not None:
                cv2.circle(frame, (int(bp["x"]), int(bp["y"])), 15, (60, 220, 60), 2)
            hud = f"BALL {bp['conf']:.2f}" if bp else "NO BALL"
            cv2.putText(frame, hud, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (60, 220, 60) if bp else (60, 60, 230), 2)
        cv2.putText(frame, f"t={now:.2f}s", (25, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)

        # 主畫面: 閃爍中的落點位置 (依可信度上色)
        for e in (bounces if "bounce" in parts else []):
            age = now - e["t"]
            if not (0 <= age < FLASH_SEC):
                continue
            if e.get("zone"):
                col, tag = (0, 255, 255), None          # 桌上: 黃
            elif e.get("table"):
                col, tag = (40, 170, 255), "OFF-TABLE"  # 桌外: 橘 (疑似擊球誤判)
            else:
                col, tag = (160, 160, 160), "UNMAPPED"  # 無桌面資訊: 灰
            x, y = int(e["x"]), int(e["y"])
            r = int(14 + 26 * (age / FLASH_SEC))
            cv2.circle(frame, (x, y), r, col, 2)
            cv2.circle(frame, (x, y), 5, col, -1)
            if tag:
                cv2.putText(frame, tag, (x + 22, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        if "strip" in parts:
            strip_right = origin[0] - 20 if args.map_pos in ("br", "bl") else w - 20
            draw_track_strip(frame, history, now, args.window, strip_right)
        if "map" in parts:
            draw_minimap(frame, [e for e in bounces if e["t"] <= now], now, origin)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"示範影片輸出於 {out_path}  (落點 {len(bounces)} 個)")


if __name__ == "__main__":
    main()
