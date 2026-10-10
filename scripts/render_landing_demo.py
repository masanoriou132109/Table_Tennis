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

from src.edge_refine import refine_quad  # noqa: E402
from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from src.video_io import (OUT_H, OUT_W, PROC_H, PROC_W, UI, check_aspect,  # noqa: E402
                          fs, lw, pt, s, to_out, to_proc)
from scripts.infer_video import predict  # noqa: E402
from scripts.landing_points import (DRIBBLE_MAX_DIST, DRIBBLE_MAX_DT,  # noqa: E402
                                    DRIBBLE_NET_TOL, mark_net_dribbles, mark_serves)

DRIBBLE_COLOR = (255, 0, 255)      # 觸網連續彈跳: 洋紅 (不計入落點)

# 以下 UI 尺寸以 720p 設計, 繪製時經 s()/fs()/lw() 放大到 1080p 輸出
TABLE_W, TABLE_H = 274.0, 152.5   # ITTF 正規球桌 (cm),物理等比
FLASH_SEC = 1.2                    # 新落點閃爍持續秒數
MAP_SCALE = 1.1                    # cm → 小視窗像素 (720p)
PAD = 16                           # 小視窗內邊距 (720p)
STRIP_H = 20                       # 底部球軌跡時間軸高度 (720p)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_track_strip(frame, history, now, window_sec, x_right):
    """底部時間軸: 最近 window_sec 秒球軌跡是否有點 (綠=有, 暗紅=無)。x_right 為 720p 座標。"""
    x0, x1 = s(20), s(x_right)
    y0 = frame.shape[0] - s(STRIP_H + 18)
    sh = s(STRIP_H)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + sh), (35, 35, 35), -1)
    t_start = now - window_sec
    for t, present in history:
        if t < t_start:
            continue
        px = x0 + int((t - t_start) / window_sec * (x1 - x0))
        cv2.line(frame, (px, y0 + s(2)), (px, y0 + sh - s(2)),
                 (60, 200, 60) if present else (40, 40, 90), lw(2))
    cv2.rectangle(frame, (x0, y0), (x1, y0 + sh), (180, 180, 180), lw(1))
    cv2.putText(frame, f"ball track  -{window_sec:.0f}s", (x0, y0 - s(6)),
                FONT, fs(0.45), (180, 180, 180), lw(1), cv2.LINE_AA)
    cv2.putText(frame, "now", (x1 - s(32), y0 - s(6)),
                FONT, fs(0.45), (180, 180, 180), lw(1), cv2.LINE_AA)


def draw_minimap(canvas, bounces, now, origin):
    """在 canvas 畫俯視桌面小視窗。bounces: 已發生的落點清單。origin 為 720p 座標。"""
    tw, th = s(TABLE_W * MAP_SCALE), s(TABLE_H * MAP_SCALE)
    pw, ph = tw + s(PAD) * 2, th + s(PAD) * 2 + s(22)
    ox, oy = pt(*origin)

    # 半透明底板
    panel = canvas[oy:oy + ph, ox:ox + pw]
    panel[:] = (panel * 0.25 + np.array([25, 25, 25]) * 0.75).astype(np.uint8)
    cv2.rectangle(canvas, (ox, oy), (ox + pw, oy + ph), (200, 200, 200), lw(1))

    tx0, ty0 = ox + s(PAD), oy + s(PAD) + s(22)
    cv2.rectangle(canvas, (tx0, ty0), (tx0 + tw, ty0 + th), (60, 95, 60), -1)
    for i in range(1, 3):  # 3x3 分區格線
        y = ty0 + int(th * i / 3)
        cv2.line(canvas, (tx0, y), (tx0 + tw, y), (110, 110, 110), lw(1))
    for half in (0, 1):
        for i in range(1, 3):
            x = tx0 + int(tw * (half * 3 + i) / 6)
            cv2.line(canvas, (x, ty0), (x, ty0 + th), (110, 110, 110), lw(1))
    cv2.rectangle(canvas, (tx0, ty0), (tx0 + tw, ty0 + th), (255, 255, 255), lw(1))
    cv2.line(canvas, (tx0 + tw // 2, ty0), (tx0 + tw // 2, ty0 + th), (0, 200, 255), lw(2))
    cv2.putText(canvas, "FAR", (tx0 + s(3), ty0 + s(12)), FONT,
                fs(0.35), (180, 180, 180), lw(1), cv2.LINE_AA)
    cv2.putText(canvas, "NEAR", (tx0 + s(3), ty0 + th - s(5)), FONT,
                fs(0.35), (180, 180, 180), lw(1), cv2.LINE_AA)

    def map_xy(e):
        # 垂直翻轉: 桌面座標 y=0 是近端,但影片中近端在畫面下方,翻轉後小視窗與影片同向
        return (tx0 + int(e["table"][0] / TABLE_W * tw),
                ty0 + th - int(e["table"][1] / TABLE_H * th))

    dribbles = [e for e in bounces if e.get("net_dribble")]
    bounces = [e for e in bounces if not e.get("net_dribble")]
    on_table = [e for e in bounces if e.get("zone")]
    off_table = [e for e in bounces if e.get("table") and not e.get("zone")]
    unmapped = [e for e in bounces if not e.get("table")]
    label = f"BOUNCES  {len(on_table)}"
    if off_table or unmapped:
        label += f"   (off {len(off_table)} / unmap {len(unmapped)})"
    if dribbles:
        label += f"   (net {len(dribbles)})"

    flash = None
    for e in on_table:
        px, py = map_xy(e)
        age = now - e["t"]
        if 0 <= age < FLASH_SEC:  # 新落點: 放大 + 擴散環
            r = s(6 + 14 * (age / FLASH_SEC))
            cv2.circle(canvas, (px, py), r, (0, 255, 255), lw(2), cv2.LINE_AA)
            cv2.circle(canvas, (px, py), s(6), (0, 255, 255), -1, cv2.LINE_AA)
            flash = (f"SERVE {e['serve']}  zone {e['zone']}" if e.get("serve")
                     else f"BOUNCE  zone {e['zone']}")
        else:                      # 歷史落點: 暗紅小點
            cv2.circle(canvas, (px, py), s(4), (90, 90, 230), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), s(4), (220, 220, 220), lw(1), cv2.LINE_AA)
    for e in off_table + unmapped:  # 可疑事件也要讓使用者看到,不靜默丟棄
        if 0 <= now - e["t"] < FLASH_SEC:
            flash = "BOUNCE  OFF-TABLE (suspect)" if e.get("table") else "BOUNCE  UNMAPPED (no table)"
    for e in dribbles:  # 觸網連續彈跳: 空心洋紅, 不計入落點, 但仍顯示供肉眼確認
        if not e.get("table"):
            continue
        px, py = map_xy(e)
        fresh = 0 <= now - e["t"] < FLASH_SEC
        cv2.circle(canvas, (px, py), s(7 if fresh else 4), DRIBBLE_COLOR,
                   lw(2 if fresh else 1), cv2.LINE_AA)
        if fresh:
            flash = "NET DRIBBLE (not counted)"

    color = (0, 255, 255) if flash and "zone" in flash else \
            (DRIBBLE_COLOR if flash and "NET" in flash else
             ((40, 170, 255) if flash else (220, 220, 220)))
    cv2.putText(canvas, flash or label, (tx0, oy + s(PAD) + s(12)),
                FONT, fs(0.5), color, lw(2), cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--window", type=float, default=12.0, help="時間軸顯示秒數")
    ap.add_argument("--parts", default="table,ball,strip,bounce,map",
                    help="要顯示的元件,逗號分隔: table,ball,strip,bounce,map")
    ap.add_argument("--map-pos", default="br", choices=("br", "tr", "bl", "tl"),
                    help="小視窗位置: br=右下 tr=右上 bl=左下 tl=左上")
    ap.add_argument("--edge-alpha", type=float, default=0.3,
                    help="edge 精修結果的 EMA 係數 (與出貨用的 infer_video 一致)")
    ap.add_argument("--dribble-max-dt", type=float, default=DRIBBLE_MAX_DT,
                    help="觸網連續彈跳: 與前一落點的最大時間差 (s)")
    ap.add_argument("--dribble-max-dist", type=float, default=DRIBBLE_MAX_DIST,
                    help="觸網連續彈跳: 與前一落點的最大桌面距離 (cm)")
    ap.add_argument("--dribble-net-tol", type=float, default=DRIBBLE_NET_TOL,
                    help="觸網連續彈跳: 前一跳離網子 <= 此距離 (cm) 時不檢查同側")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("START", "END"), default=None,
                    help="只輸出這段 (秒)。前 5 秒先跑追蹤暖機但不輸出; 小視窗只顯示片段內的落點")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}

    data = json.loads(Path(args.json_path).read_text())
    # 渲染時重新判定 (舊 JSON 也適用, 調門檻不必重跑球偵測)
    n_dribble = mark_net_dribbles(data["events"], args.dribble_max_dt, args.dribble_max_dist,
                                  args.dribble_net_tol)
    print(f"觸網連續彈跳: {n_dribble} 個 "
          f"(Δt<={args.dribble_max_dt}s, Δd<={args.dribble_max_dist}cm, 同側或前跳離網<={args.dribble_net_tol}cm, 中間無擊球)")
    n_serve = mark_serves(data["events"])
    print(f"發球: 兩跳都抓到 {n_serve[1]} 次, 只抓到一跳/無法確定 {n_serve['?']} 次")
    # 全部落點都要顯示: 桌上(正常) / 桌外(疑似擊球誤判) / 未映射(當時無桌面資訊)。
    # 先前只取有 table 的,導致未映射的落點在 demo 中完全消失。
    bounces = [e for e in data["events"] if e.get("type") == "bounce"]
    video, fps = data["video"], data["fps"]
    start, dur = data["start"], data["dur"]
    write_from = start
    if args.clip:
        # 追蹤器的持久先驗需要先看過完整桌面, 冷啟動會沒有框 → 提前 5 秒暖機
        warm = max(start, args.clip[0] - 5.0)
        dur = min(start + dur, args.clip[1]) - warm
        start, write_from = warm, args.clip[0]
        bounces = [e for e in bounces if args.clip[0] <= e["t"] <= args.clip[1]]
    # 球軌跡: 實際餵給落點演算法的點 (JSON 內已有,不需重跑球偵測)
    track_by_frame = {p["frame"]: p for p in data.get("track", [])}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(video)
    check_aspect(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 偵測/追蹤在 1280x720 (與 JSON 座標一致), 畫面輸出 1080p
    w, h = PROC_W, PROC_H
    start_f = int(start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tracker = TableTracker(w, h)

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"demo_{Path(video).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))

    # 版面以 720p 座標規劃, 繪製時放大
    map_w = int(TABLE_W * MAP_SCALE) + PAD * 2
    map_h = int(TABLE_H * MAP_SCALE) + PAD * 2 + 22
    M = 20
    origin = {"br": (w - map_w - M, h - map_h - M),
              "tr": (w - map_w - M, M),
              "bl": (M, h - map_h - M),
              "tl": (M, M)}[args.map_pos]

    history: list[tuple[float, bool]] = []
    edge_state = None   # 精修結果的 EMA 狀態 (抑制逐幀抖動)
    for i in range(int(dur * fps)):
        ok, raw = cap.read()
        if not ok:
            break
        fi = start_f + i
        now = fi / fps
        frame = to_proc(raw)
        canvas = to_out(raw) if now >= write_from else None   # 暖機期不需要畫

        coords, scores, presence = predict(model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        # 桌面框走與出貨相同的管線: 追蹤器 → edge 精修 → EMA。只畫矩形框,
        # 不畫角點或信心 (此 demo 的重點是落點, 桌面只是參考框)。
        if quad is None:
            edge_state = None
        else:
            refined, _, _ = refine_quad(frame, quad, tracker.smoother.confident,
                                        reference=tracker.smoother.reference)
            edge_state = refined if edge_state is None else \
                (1 - args.edge_alpha) * edge_state + args.edge_alpha * refined

        bp = track_by_frame.get(fi)
        history.append((now, bp is not None))
        if canvas is None:
            continue

        if "table" in parts and edge_state is not None:
            cv2.polylines(canvas, [np.round(edge_state * UI).astype(np.int32)],
                          True, (0, 255, 0), lw(2), cv2.LINE_AA)

        # 球 (軌跡點)
        if "ball" in parts:
            if bp is not None:
                cv2.circle(canvas, pt(bp["x"], bp["y"]), s(15), (60, 220, 60), lw(2), cv2.LINE_AA)
            hud = f"BALL {bp['conf']:.2f}" if bp else "NO BALL"
            cv2.putText(canvas, hud, pt(25, 45), FONT, fs(0.9),
                        (60, 220, 60) if bp else (60, 60, 230), lw(2), cv2.LINE_AA)
        cv2.putText(canvas, f"t={now:.2f}s", pt(25, 75), FONT,
                    fs(0.6), (200, 200, 200), lw(1), cv2.LINE_AA)

        # 主畫面: 閃爍中的落點位置 (依可信度上色)
        for e in (bounces if "bounce" in parts else []):
            age = now - e["t"]
            if not (0 <= age < FLASH_SEC):
                continue
            if e.get("net_dribble"):
                col, tag = DRIBBLE_COLOR, "NET DRIBBLE"  # 觸網連續彈跳: 洋紅 (不計入)
            elif e.get("zone"):
                col, tag = (0, 255, 255), None          # 桌上: 黃
            elif e.get("table"):
                col, tag = (40, 170, 255), "OFF-TABLE"  # 桌外: 橘 (疑似擊球誤判)
            else:
                col, tag = (160, 160, 160), "UNMAPPED"  # 無桌面資訊: 灰
            if e.get("serve") and not e.get("net_dribble"):
                tag = f"SERVE {e['serve']}" + (f" {tag}" if tag else "")
            x, y = e["x"], e["y"]
            r = 14 + 26 * (age / FLASH_SEC)
            cv2.circle(canvas, pt(x, y), s(r), col, lw(2), cv2.LINE_AA)
            cv2.circle(canvas, pt(x, y), s(5), col, -1, cv2.LINE_AA)
            if tag:
                cv2.putText(canvas, tag, pt(x + 22, y - 10), FONT, fs(0.55), col, lw(2), cv2.LINE_AA)

        if "strip" in parts:
            strip_right = origin[0] - 20 if args.map_pos in ("br", "bl") else w - 20
            draw_track_strip(canvas, history, now, args.window, strip_right)
        if "map" in parts:
            draw_minimap(canvas, [e for e in bounces if e["t"] <= now], now, origin)
        writer.write(canvas)

    writer.release()
    cap.release()
    print(f"示範影片輸出於 {out_path}  (落點 {len(bounces)} 個)")


if __name__ == "__main__":
    main()
