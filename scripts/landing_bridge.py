#!/usr/bin/env python3
"""給其他程式用的橋接: 拿它自己的桌面框 + 球偵測, 跑我們的落點分析。

為 analyze (8).py (半自動比賽記錄工具, 教授 repo 的 add-analysis-script 分支) 寫的。
它已經自己跑過 TableTracker 和球偵測, 這裡直接沿用那些資料, 不重跑模型:

    from scripts.landing_bridge import detect_landings
    bounces = detect_landings(quads_native, dets_boxes, fps, vid_w, vid_h, TABLE_W, TABLE_H)

輸出是它的格式 (影片原始像素 + 它的圖表座標, y=0 在遠端), 並依我們的規則篩過:
只留「在桌上且算數」的落點 (含發球第 1/2 跳; 它判斷發球者需要發球第 1 跳),
own_side / second_bounce / net_edge / dead_ball / 桌外的都不給。
include_uncounted=True 則全部給, 原因放在 "uncounted"。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.landing_points import (DEFAULT_PROF_REPO, TABLE_H_CM, TABLE_W_CM,  # noqa: E402
                                    analyze_frames, build_parser, load_prof_module)
from src.video_io import PROC_H, PROC_W  # noqa: E402

_de = None


def _prof_module():
    """教授模組只載入一次。"""
    global _de
    if _de is None:
        _de = load_prof_module(Path(DEFAULT_PROF_REPO))
    return _de


def detect_landings(quads_native: dict, dets_boxes: dict, fps: float, vid_w: int, vid_h: int,
                    chart_w: float, chart_h: float, video: str = "",
                    include_uncounted: bool = False) -> list[dict]:
    """
    quads_native: 幀 → TableTracker 的四邊形 [FL,FR,NR,NL] (影片原始像素); 沒框的幀不放
    dets_boxes:   幀 → [[x1,y1,x2,y2,conf], ...] (影片原始像素); 每一幀都要有 key (可為空 list)
    chart_w/h:    呼叫端圖表座標的大小 (桌面畫成 chart_w x chart_h, FL 在 (0,0))
    回傳 [{"t","frame","x","y","tx","ty","uncounted","serve","table_cm"}], 依時間排序。
    """
    de = _prof_module()
    sx, sy = vid_w / PROC_W, vid_h / PROC_H
    fps = fps or 30.0

    # 我們的門檻都在 1280x720 校準: 先換成處理座標
    frames = [{"t": fi / fps, "frame": fi,
               "dets": [{"x": (b[0] + b[2]) / 2 / sx, "y": (b[1] + b[3]) / 2 / sy, "conf": b[4]}
                        for b in dets_boxes[fi]]}
              for fi in sorted(dets_boxes)]
    quads = {fi: np.asarray(q, float) / [sx, sy] for fi, q in quads_native.items()}
    dur = (max(dets_boxes) + 1) / fps if dets_boxes else 0.0
    args = build_parser().parse_args([video or "_", "--start", "0", "--dur", f"{dur:g}"])
    result, _ = analyze_frames(frames, quads, fps, [vid_w, vid_h], args, de)

    out = []
    for e in result["events"]:
        if e.get("type") != "bounce" or not e.get("table"):
            continue
        if not include_uncounted and not (e.get("zone") and not e.get("uncounted")):
            continue
        cx, cy = e["table"]
        out.append({"t": e["t"], "frame": e["t"] * fps, "x": e["x"] * sx, "y": e["y"] * sy,
                    # 教授的桌面座標 y=0 在近端; 呼叫端的圖表 y=0 在遠端
                    "tx": cx * chart_w / TABLE_W_CM, "ty": (TABLE_H_CM - cy) * chart_h / TABLE_H_CM,
                    "uncounted": e.get("uncounted"), "serve": e.get("serve"), "table_cm": [cx, cy]})
    return out
