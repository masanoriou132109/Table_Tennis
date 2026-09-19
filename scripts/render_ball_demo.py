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

from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from scripts.infer_video import predict  # noqa: E402
from scripts.landing_points import (BALL_INPUT, TABLE_W_CM, ball_detections,  # noqa: E402
                                    load_prof_module)

STRIP_H = 22


def draw_strip(frame, history, now, window_sec):
    """底部滾動時間軸: history = [(t, state)], state 0=無 1=有 2=被閘門擋."""
    h, w = frame.shape[:2]
    x0, x1 = 20, w - 20
    y0 = h - STRIP_H - 14
    cv2.rectangle(frame, (x0, y0), (x1, y0 + STRIP_H), (35, 35, 35), -1)
    t_start = now - window_sec
    colors = {0: (40, 40, 90), 1: (60, 200, 60), 2: (40, 180, 200)}
    for t, st in history:
        if t < t_start:
            continue
        px = x0 + int((t - t_start) / window_sec * (x1 - x0))
        cv2.line(frame, (px, y0 + 2), (px, y0 + STRIP_H - 2), colors[st], 2)
    cv2.rectangle(frame, (x0, y0), (x1, y0 + STRIP_H), (180, 180, 180), 1)
    cv2.putText(frame, f"-{window_sec:.0f}s", (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, "now", (x1 - 30, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--prof-repo", default=str(PROJECT_ROOT / "external" / "PingPongTracker"))
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
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
    import coremltools as ct
    ball_model = ct.models.MLModel(str(repo / "Models" / "BallDetector.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    table_model = TableKeypointNet(pretrained=False).to(device)
    table_model.load_state_dict(ckpt["model"])
    table_model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_f = int(args.start * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    tracker = TableTracker(w, h)

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"balldemo_{Path(args.video).stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    history: list[tuple[float, int]] = []
    table_ok: list[tuple[float, bool]] = []
    for i in range(int(args.dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        now = (start_f + i) / fps

        coords, scores, presence = predict(table_model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        H = None
        if quad is not None:
            cv2.polylines(frame, [quad.astype(np.int32)], True, (0, 200, 0), 2)
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
            cv2.circle(frame, (int(best["x"]), int(best["y"])), 16, col, 2)
            cv2.putText(frame, f"{best['conf']:.2f}",
                        (int(best["x"]) + 20, int(best["y"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        history.append((now, state))

        hud = f"BALL {best['conf']:.2f}" if state == 1 else \
              ("OFF-TABLE DET" if state == 2 else "NO BALL")
        hud_col = (60, 220, 60) if state == 1 else ((40, 200, 220) if state == 2 else (60, 60, 230))
        cv2.putText(frame, hud, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_col, 2)
        cv2.putText(frame, f"t={now:.2f}s", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)
        draw_strip(frame, history, now, args.window)
        writer.write(frame)

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
