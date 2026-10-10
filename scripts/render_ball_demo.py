#!/usr/bin/env python3
"""輸出「球偵測檢視」影片: 標出偵測到的球,並用滾動時間軸顯示漏偵測的空隙。

畫面:
  - 綠圈 + 信心值: 該幀偵測到球 (且通過桌面區域閘門)
  - 黃圈: 偵測到但被場外閘門擋掉 (可能是誤判)
  - 左上 HUD: BALL <conf> (綠) / NO BALL (紅)
  - 底部時間軸: 最近 --window 秒的偵測狀態 (綠=有, 暗紅=無), 讓空隙一目了然

同時在終端印出「桌面可見但連續數幀沒偵測到球」的時段清單,方便跳轉檢視。

用法:
    python scripts/render_ball_demo.py "videos/xxx.mp4" --start 200 --dur 40
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

from src.tracker import TableTracker  # noqa: E402
from src.video_io import (OUT_H, OUT_W, PROC_H, PROC_W, UI, check_aspect,  # noqa: E402
                          fs, lw, pt, s, to_out, to_proc)
from scripts.infer_video import load_table_model, predict  # noqa: E402
from scripts.landing_points import (DEFAULT_PROF_REPO, TABLE_W_CM, BallModel,  # noqa: E402
                                    ball_detections, load_prof_module)

STRIP_H = 22


def draw_strip(frame, history, now, window_sec):
    """底部滾動時間軸: history = [(t, state)], state 0=無 1=有 2=被閘門擋. 尺寸以 720p 設計後放大."""
    h, w = frame.shape[:2]
    x0, x1 = s(20), w - s(20)
    y0 = h - s(STRIP_H + 14)
    sh = s(STRIP_H)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + sh), (35, 35, 35), -1)
    t_start = now - window_sec
    colors = {0: (40, 40, 90), 1: (60, 200, 60), 2: (40, 180, 200)}
    for t, st in history:
        if t < t_start:
            continue
        px = x0 + int((t - t_start) / window_sec * (x1 - x0))
        cv2.line(frame, (px, y0 + s(2)), (px, y0 + sh - s(2)), colors[st], lw(2))
    cv2.rectangle(frame, (x0, y0), (x1, y0 + sh), (180, 180, 180), lw(1))
    cv2.putText(frame, f"-{window_sec:.0f}s", (x0, y0 - s(5)),
                cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), (180, 180, 180), lw(1), cv2.LINE_AA)
    cv2.putText(frame, "now", (x1 - s(30), y0 - s(5)),
                cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), (180, 180, 180), lw(1), cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--prof-repo", default=str(DEFAULT_PROF_REPO))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--table-backend", choices=("coreml", "torch"), default="coreml",
                    help="桌面模型執行方式: coreml (預設, Models/TableDetector.mlpackage) / torch (checkpoint)")
    ap.add_argument("--ball-model", default=None,
                    help="球偵測 .mlpackage (預設: Models/BallDetector_rfdetr_20260930.mlpackage, "
                         "RF-DETR Large)。YOLO / RF-DETR 格式自動判斷")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--zone-expand", type=float, default=0.4 * TABLE_W_CM)
    ap.add_argument("--window", type=float, default=12.0, help="時間軸顯示秒數")
    ap.add_argument("--min-gap", type=float, default=0.3, help="列為漏偵測的最短空隙秒數")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(args.prof_repo)
    de = load_prof_module(repo)
    ball_model = BallModel(args.ball_model)
    print(ball_model)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    table_model = load_table_model(args.ckpt, device, args.table_backend)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    check_aspect(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w, h = PROC_W, PROC_H   # 偵測在 1280x720, 畫面輸出 1080p
    start_f = int(args.start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tracker = TableTracker(w, h)

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"balldemo_{Path(args.video).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))

    history: list[tuple[float, int]] = []
    table_ok: list[tuple[float, bool]] = []
    for i in range(int(args.dur * fps)):
        ok, raw = cap.read()
        if not ok:
            break
        now = (start_f + i) / fps
        frame, canvas = to_proc(raw), to_out(raw)

        coords, scores, presence = predict(table_model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        H = None
        if quad is not None:
            cv2.polylines(canvas, [np.round(quad * UI).astype(np.int32)], True, (0, 200, 0), lw(2),
                          cv2.LINE_AA)
            H = de.build_homography(quad[::-1].astype(np.float32))
        table_ok.append((now, quad is not None))

        dets = ball_detections(ball_model, frame, args.conf)

        def in_gate(d):
            if H is None:
                return True
            tx, ty = de.map_point(H, d["x"], d["y"])
            return (-args.zone_expand <= tx <= de.TABLE_W + args.zone_expand
                    and -args.zone_expand <= ty <= de.TABLE_H + args.zone_expand)

        inside = [d for d in dets if in_gate(d)]
        if inside:                       # 通過閘門 → 視為有效偵測
            best, state = max(inside, key=lambda d: d["conf"]), 1
        elif dets:                       # 只有場外偵測 → 可能是誤判
            best, state = max(dets, key=lambda d: d["conf"]), 2
        else:
            best, state = None, 0
        if best is not None:
            col = (60, 220, 60) if state == 1 else (40, 200, 220)
            cv2.circle(canvas, pt(best["x"], best["y"]), s(16), col, lw(2), cv2.LINE_AA)
            cv2.putText(canvas, f"{best['conf']:.2f}", pt(best["x"] + 20, best["y"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs(0.6), col, lw(2), cv2.LINE_AA)
        history.append((now, state))

        hud = f"BALL {best['conf']:.2f}" if state == 1 else \
              ("OFF-TABLE DET" if state == 2 else "NO BALL")
        hud_col = (60, 220, 60) if state == 1 else ((40, 200, 220) if state == 2 else (60, 60, 230))
        cv2.putText(canvas, hud, pt(25, 45), cv2.FONT_HERSHEY_SIMPLEX, fs(0.9), hud_col, lw(2),
                    cv2.LINE_AA)
        cv2.putText(canvas, f"t={now:.2f}s", pt(25, 75), cv2.FONT_HERSHEY_SIMPLEX, fs(0.6),
                    (200, 200, 200), lw(1), cv2.LINE_AA)
        draw_strip(canvas, history, now, args.window)
        writer.write(canvas)

    writer.release()
    cap.release()

    # 桌面可見但沒偵測到球的連續時段
    tmap = dict(table_ok)
    gaps, run_start = [], None
    for t, st in history:
        missing = (st != 1) and tmap.get(t, False)
        if missing and run_start is None:
            run_start = t
        elif not missing and run_start is not None:
            if t - run_start >= args.min_gap:
                gaps.append((run_start, t))
            run_start = None
    if run_start is not None and history[-1][0] - run_start >= args.min_gap:
        gaps.append((run_start, history[-1][0]))

    n_det = sum(1 for _, s in history if s == 1)
    n_tab = sum(1 for _, ok in table_ok if ok)
    print(f"影格 {len(history)}  桌面可見 {n_tab}  偵測到球 {n_det} "
          f"(桌面可見時的覆蓋率 {n_det/max(n_tab,1):.0%})")
    print(f"\n桌面可見但未偵測到球、長度 >= {args.min_gap}s 的時段 ({len(gaps)} 段):")
    for a, b in gaps:
        print(f"  {a:8.2f}s ~ {b:8.2f}s   ({b-a:.2f}s)")
    print(f"\n影片輸出於 {out_path}")


if __name__ == "__main__":
    main()
