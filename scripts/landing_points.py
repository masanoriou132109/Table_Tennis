#!/usr/bin/env python3
"""端到端落點分析: 本專案的逐幀桌角 + PingPongTracker 的球偵測與事件演算法。

串接方式 (完全不修改教授的程式碼):
  1. 本專案 TableKeypointNet + TableTracker → 每幀桌面 4 角 → 每幀 homography
  2. 教授的球偵測 Core ML (預設 RF-DETR Large, Models/BallDetector_rfdetr_20260930.mlpackage;
     --ball-model 可換回 YOLO26 等其他匯出, 格式自動判斷) → 每幀球偵測
  3. 教授 Tools/detect_events.py 的 build_track / detect_events (傳 H=None)
  4. 落點事件用「該事件時刻的那一幀」的 H 映射成桌面座標與分區

為何要逐幀 H: 他原本的 --corners 是整支影片一組固定角點,只適用固定機位;
轉播有 zoom/pan 時固定角點會讓落點映射失準。本腳本改為逐幀,
故 zoom/pan 下映射仍正確 — 這正是本專案桌面偵測的價值所在。

角點順序: 本專案為 far_left, far_right, near_right, near_left;
他的為 near-left, near-right, far-right, far-left — 兩者互為反序。

用法:
    python scripts/landing_points.py "videos/xxx.mp4" --start 200 --dur 40
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

# 教授 repo 的位置: 本專案作為子資料夾放在 PingPongTracker 內時 → 上一層;
# 獨立 repo 時 → external/PingPongTracker (另行 clone)
DEFAULT_PROF_REPO = (PROJECT_ROOT.parent
                     if (PROJECT_ROOT.parent / "Tools" / "detect_events.py").exists()
                     else PROJECT_ROOT / "external" / "PingPongTracker")

# 預設球偵測模型: 教授 2026-09-30 提供的 RF-DETR Large (教授指定改用新模型)。
# .mlpackage 不在 git 內, 需先用 scripts/export_ball_rfdetr.py 轉出。
# 舊的 YOLO26 仍可用 --ball-model <教授 repo>/Models/BallDetector.mlpackage 指定。
DEFAULT_BALL_MODEL = PROJECT_ROOT / "Models" / "BallDetector_rfdetr_20260930.mlpackage"

from src.tracker import TableTracker  # noqa: E402
from src.video_io import PROC_H, PROC_W, check_aspect, to_proc  # noqa: E402
from scripts.infer_video import load_table_model, predict  # noqa: E402

# ITTF 正規球桌尺寸 (cm)。教授原本 detect_events.py 寫 500x240 (長寬比 2.08),
# 與真實球桌 (274/152.5 = 1.80) 不符 — 他已確認是筆誤。
# 分區判定用的是比例,不受影響;但落點座標的物理意義與「離邊線幾公分」等量測需要正確尺寸。
TABLE_W_CM, TABLE_H_CM = 274.0, 152.5


def load_prof_module(repo: Path):
    """匯入教授的 detect_events.py (頂層只依賴 cv2/numpy,可安全 import)。"""
    tools = repo / "Tools"
    if not (tools / "detect_events.py").exists():
        raise FileNotFoundError(f"找不到 {tools/'detect_events.py'}")
    sys.path.insert(0, str(tools))
    import detect_events as de  # noqa: E402
    # 修正桌面尺寸 (教授確認原 500x240 為筆誤)。他的 build_homography / zone_of
    # 都讀模組全域變數,所以在此覆寫即可讓整條管線使用正確尺寸,無需改他的檔案。
    de.TABLE_W, de.TABLE_H = TABLE_W_CM, TABLE_H_CM
    return de


def patch_debounce(de):
    """修正: find_reversals 的去抖動不分方向,落點會被附近的「頂點」壓過。

    他的流程是「先去抖動(只留最強的)→ 再過濾方向」。球的最高點 (升→降) 速度通常
    比反彈 (降→升) 大,於是在 EVENT_DEBOUNCE 窗內把真正的落點吸收掉,方向過濾後
    整個群集變成空的。實測 Doha 有 3 個落點因此消失。

    改為「先過濾方向 → 再去抖動」。只影響 key=='y' (落點);擊球 (key=='x') 不變。
    """
    orig = de.find_reversals

    def find_reversals_dir_first(run, key, min_speed):
        if key != "y":
            return orig(run, key, min_speed)
        cands = []
        for i in range(1, len(run) - 1):
            bf, af = de.windowed_slope(run, i, key)
            if bf is None or af is None:
                continue
            if bf > 0 > af and abs(bf) > min_speed and abs(af) > min_speed:
                cands.append((i, abs(bf) + abs(af)))
        kept = []
        for i, strength in cands:
            if kept:
                j = kept[-1][0]
                close_t = run[i]["t"] - run[j]["t"] < de.EVENT_DEBOUNCE
                close_p = np.hypot(run[i]["x"] - run[j]["x"],
                                   run[i]["y"] - run[j]["y"]) < de.EVENT_DEBOUNCE_DIST
                if close_t and close_p:
                    if strength > kept[-1][1]:
                        kept[-1] = (i, strength)
                    continue
            kept.append((i, strength))
        return [i for i, _ in kept]

    de.find_reversals = find_reversals_dir_first


class BallModel:
    """Core ML 球偵測器 + 其輸出格式。支援教授的兩種匯出:

    - yolo:   YOLO26 end2end, 單一 [1,N,6] = x1,y1,x2,y2,conf,cls (輸入像素座標)
    - rfdetr: RF-DETR, pred_boxes [1,Q,4] 正規化 cxcywh + pred_logits [1,Q,C]
              (無 NMS; 分數取各類別 sigmoid 最大值, 與 PingPongBallDetector.swift 相同)

    格式與輸入邊長由模型本身判斷 (輸出名稱 / input spec), 不需手動指定。
    """

    def __init__(self, path: str | Path | None = None):
        import coremltools as ct
        path = Path(path) if path else DEFAULT_BALL_MODEL
        if not path.exists():
            raise SystemExit(
                f"找不到球偵測模型 {path}\n"
                "請先轉出 RF-DETR 模型:\n"
                "  ~/Developer/tt-export-venv311/bin/python scripts/export_ball_rfdetr.py "
                "checkpoints/ball/pingpong20260930_rfdetr.pt --model-class RFDETRLarge "
                f"--out {DEFAULT_BALL_MODEL.relative_to(PROJECT_ROOT)}\n"
                "或用 --ball-model 指定其他模型 (例如教授 repo 的 Models/BallDetector.mlpackage)")
        spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
        outs = {o.name for o in spec.description.output}
        self.arch = "rfdetr" if "pred_boxes" in outs else "yolo"
        self.imgsz = int(spec.description.input[0].type.imageType.width)
        # YOLO 匯出在 GPU 路徑會 crash (教授註解), 只能 CPU;
        # RF-DETR 教授已驗證各 compute unit 皆可, 用 ALL 取得 ANE/GPU 速度。
        cu = ct.ComputeUnit.ALL if self.arch == "rfdetr" else ct.ComputeUnit.CPU_ONLY
        self.ml = ct.models.MLModel(str(path), compute_units=cu)
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"BallModel({self.path.name}, {self.arch} @ {self.imgsz})"


TOUCH_MAX_GAP_F = 3    # 夾住觸桌點的兩個觀察點最多相隔幾幀 (超過就保留教授的估計)
TOUCH_SIDE_N = 4       # 觸桌點兩側各用幾個點擬合 y(t)
TOUCH_SIDE_DT = 0.15   # s; 擬合點離夾住的觀察點最遠多久


def _fit_t(pts: list[dict], key: str, deg: int):
    ts = np.array([p["t"] for p in pts])
    vs = np.array([p[key] for p in pts])
    return np.polyfit(ts - ts[0], vs, min(deg, len(pts) - 1)), ts[0]


def _eval_t(fit, t: float) -> float:
    p, t0 = fit
    return float(np.polyval(p, t - t0))


def refine_touch_points(events: list[dict]) -> int:
    """用時間參數化重估觸桌點 (ev["x"], ev["y"]), 教授的估計存在 ev["xy_prof"]。

    教授的 refine_bounce 用 y(x) 拋物線交點, 兩種情況會偏:
      - 斜率視窗選到的 V 點不是最低點, 觸桌後的點被算進入射段 (London 28:23.18, 偏 46px)
      - 球幾乎垂直落下, y(x) 病態 (webm 97.32, 偏 9px → 落點被映射到網子另一側)
    做法: V 點附近 ±2 點內找觀察到的最低點, 觸桌一定在它與前一點或後一點之間。
    兩種夾法都試: 兩側各擬合 y(t), 在兩個觀察點的時間區間內找交點 t*,
    x 用兩點線性內插 → 結果不會跑出這兩個觀察點之間。取兩段 y(t) 最接近的夾法。
    偵測斷掉 (夾住的兩點相隔 > TOUCH_MAX_GAP_F 幀) 或點不夠時保留教授的估計。
    連續相同位置的點 (重複幀影片, 例如 25fps 補成 50fps) 先去掉, 否則 y(t) 會被拉平。
    回傳改用新估計的數量。
    """
    n = 0
    for e in events:
        if e.get("type") != "bounce" or "xy_prof" in e:
            continue
        legs = e.get("debug") or {}
        l1, l2 = legs.get("leg1_points") or [], legs.get("leg2_points") or []
        if not l1 or not l2:
            continue
        by_frame = {p["frame"]: p for p in l1 + l2}
        pts: list[dict] = []
        for p in sorted(by_frame.values(), key=lambda p: p["t"]):
            if pts and abs(p["x"] - pts[-1]["x"]) < 0.5 and abs(p["y"] - pts[-1]["y"]) < 0.5:
                continue
            pts.append(p)
        vi = min(range(len(pts)), key=lambda i: abs(pts[i]["frame"] - l1[-1]["frame"]))
        if e.get("kink"):
            m = vi            # 折角落點沒有 y 最低點, 直接夾住折角那一點
        else:
            m = max(range(max(0, vi - 2), min(len(pts) - 1, vi + 2) + 1), key=lambda i: pts[i]["y"])

        best = None   # (y(t) 差距, x, y)
        for a in (m - 1, m):                  # 觸桌在 pts[a] 與 pts[a+1] 之間
            if a < 0 or a + 1 >= len(pts):
                continue
            P, Q = pts[a], pts[a + 1]
            if Q["frame"] - P["frame"] > TOUCH_MAX_GAP_F:
                continue
            A = [p for p in pts[max(0, a - TOUCH_SIDE_N + 1):a + 1] if P["t"] - p["t"] <= TOUCH_SIDE_DT]
            B = [p for p in pts[a + 1:a + 1 + TOUCH_SIDE_N] if p["t"] - Q["t"] <= TOUCH_SIDE_DT]
            if len(A) < 2 or len(B) < 2:
                continue
            yA, yB = _fit_t(A, "y", 2), _fit_t(B, "y", 2)
            ts = np.linspace(P["t"], Q["t"], 101)
            gap = np.abs([_eval_t(yA, t) - _eval_t(yB, t) for t in ts])
            k = int(np.argmin(gap))
            f = (ts[k] - P["t"]) / (Q["t"] - P["t"])
            x = P["x"] + f * (Q["x"] - P["x"])
            # y 外插可能爆掉: 限制在兩點之間到「低於較低點一個兩點距離」以內
            y = (_eval_t(yA, ts[k]) + _eval_t(yB, ts[k])) / 2
            y_hi = max(P["y"], Q["y"]) + float(np.hypot(Q["x"] - P["x"], Q["y"] - P["y"]))
            y = min(max(y, min(P["y"], Q["y"])), y_hi)
            if best is None or gap[k] < best[0]:
                best = (float(gap[k]), float(x), float(y))
        if best is None:
            continue
        e["xy_prof"] = [e["x"], e["y"], e["method"]]
        e["x"], e["y"], e["method"] = best[1], best[2], "touch-t"
        n += 1
    return n


KINK_N = 3             # 折角前後各用幾個點估 y 速度
KINK_MAX_GAP_F = 3     # 這些點之間最多相隔幾幀
KINK_DV_FRAC = 2.2     # y 速度往上突變 >= 此倍數 x 桌面畫面高度 (每秒) 才算 (London 桌高 112px → 250px/s)
KINK_EXCLUDE_S = 0.1   # 離擊球或既有落點這麼近的折角不算 (擊球本身也是折角)
KINK_MIN_STEP = 2.0    # px; 每一步都要移動超過這麼多 (排除拿在手上/拋球、在兩個固定位置跳動的雜訊)


def detect_kink_bounces(track: list[dict], events: list[dict], table_px_h: float) -> int:
    """補抓「沒有 y 最低點」的落點: 球往遠端飛時, 下落被往遠端的移動抵消, y 速度一直 <= 0,
    教授的「y 速度由正轉負」抓不到 (London 0:17.02、Lin vs Moregard 2:31.61)。

    反彈一定讓畫面上的 y 速度突然往上改變, 不管球往哪個方向飛。London 統計:
    已偵測落點的變化 -773 ~ -119 px/s (56 個中 54 個 <= -220), 飛行段 p5 = -44。
    門檻依桌面在畫面中的高度縮放 (同 MIN_Y_SPEED)。只補「教授沒抓到的」, 既有落點不動。
    新落點標 ev["kink"]=True, 觸桌點交給 refine_touch_points。回傳新增數量。
    """
    k = KINK_N
    thr = -KINK_DV_FRAC * table_px_h
    busy = [e["t"] for e in events if e.get("type") in ("hit", "bounce")]

    def vel(seg):
        return np.polyfit([p["t"] for p in seg], [p["y"] for p in seg], 1)[0]

    cands = []   # (i, dv)
    for i in range(k - 1, len(track) - k + 1):
        seg = track[i - k + 1:i + k]
        steps = list(zip(seg[:-1], seg[1:]))
        if any(q["frame"] - p["frame"] > KINK_MAX_GAP_F for p, q in steps):
            continue
        if any(np.hypot(q["x"] - p["x"], q["y"] - p["y"]) <= KINK_MIN_STEP for p, q in steps):
            continue
        if any(abs(track[i]["t"] - t) <= KINK_EXCLUDE_S for t in busy):
            continue
        dv = vel(track[i:i + k]) - vel(track[i - k + 1:i + 1])
        if dv <= thr:
            cands.append((i, dv))

    # 相鄰幾點都過門檻時 (同一個折角) 取變化最大的
    picked: list[tuple[int, float]] = []
    for i, dv in cands:
        if picked and track[i]["t"] - track[picked[-1][0]]["t"] <= KINK_EXCLUDE_S:
            if dv < picked[-1][1]:
                picked[-1] = (i, dv)
            continue
        picked.append((i, dv))

    for i, dv in picked:
        p = track[i]
        leg1, leg2 = track[max(0, i - TOUCH_SIDE_N):i + 1], track[i:i + TOUCH_SIDE_N + 1]
        dbg = lambda seg: [{"t": q["t"], "frame": q["frame"], "x": q["x"], "y": q["y"]} for q in seg]
        events.append({"type": "bounce", "t": p["t"], "frame": p["frame"], "x": p["x"], "y": p["y"],
                       "method": "kink", "leg_sizes": [len(leg1), len(leg2)], "fit_gap_px": None,
                       "kink": True, "kink_dv": round(float(dv), 1),
                       "debug": {"leg1_points": dbg(leg1), "leg2_points": dbg(leg2),
                                 "fit1": None, "fit2": None}})
    events.sort(key=lambda e: e["t"])
    return len(picked)


EDGE_BUFFER_CM = 5.0   # 超出桌邊 <= 此距離的落點仍算桌上 (擦邊球)


def apply_edge_buffer(events: list[dict], buf: float = EDGE_BUFFER_CM) -> int:
    """超出桌邊 buf 以內的落點拉回桌邊並給分區 (ev["edge_buffer"]=True, 原座標存 table_raw)。

    教授的 zone_of 完全沒有緩衝, 超出 0.1cm 就算桌外。5cm = 球半徑 2cm + 桌角定位
    誤差 (約 2px; 深度方向被透視壓縮, 1px 約 1.5cm)。實例 London 28:20.72 擦邊落點
    超出 2.4cm。超出 7.5cm 的網帶 (Lin 02:32.03) 與擊球誤判 (London 56.46) 維持桌外。
    可重複呼叫 (以 table_raw 為準)。回傳拉回的數量。
    """
    n = 0
    for e in events:
        if e.get("type") != "bounce":
            continue
        raw = e.get("table_raw") or e.get("table")
        if not raw:
            continue
        x, y = raw
        dx, dy = max(-x, x - TABLE_W_CM, 0.0), max(-y, y - TABLE_H_CM, 0.0)
        if dx == 0 and dy == 0:
            continue                            # 本來就在桌上
        e.pop("edge_buffer", None)
        if (dx * dx + dy * dy) ** 0.5 <= buf:
            cx, cy = min(max(x, 0.0), TABLE_W_CM), min(max(y, 0.0), TABLE_H_CM)
            e["table_raw"], e["table"], e["edge_buffer"] = raw, [round(cx, 1), round(cy, 1)], True
            e["zone"] = zone_of_cm(cx, cy)
            n += 1
        elif "table_raw" in e:                  # 門檻調小後還原
            e["table"], e["zone"] = raw, None
            del e["table_raw"]
    return n


def zone_of_cm(tx: float, ty: float) -> str:
    """與教授 zone_of 相同的分區 (半場 L/R + 3x3), 輸入須在桌面內。"""
    half = "L" if tx < TABLE_W_CM / 2 else "R"
    hx = tx if half == "L" else tx - TABLE_W_CM / 2
    col = min(int(hx / (TABLE_W_CM / 2 / 3)), 2)
    row = min(int(ty / (TABLE_H_CM / 3)), 2)
    return f"{half}{row * 3 + col + 1}"


DRIBBLE_MAX_DT = 0.6      # s
DRIBBLE_MAX_DIST = 60.0   # cm (桌面座標)
DRIBBLE_NET_TOL = 10.0    # cm: 前一跳離網子這麼近 → 位置誤差下分不出哪一側, 不檢查同側


def mark_net_dribbles(events: list[dict], max_dt: float = DRIBBLE_MAX_DT,
                      max_dist: float = DRIBBLE_MAX_DIST,
                      net_tol: float = DRIBBLE_NET_TOL) -> int:
    """標記觸網後在桌上連續彈跳的「多餘落點」(ev["net_dribble"]=True), 不刪除。

    條件 (全部成立才標): 與前一個落點同一側 AND 桌面距離 <= max_dist AND
    時間差 <= max_dt AND 兩者之間沒有擊球。保留第一跳, 只標後續的跳。

    同側的例外: 前一跳離網子 <= net_tol 時不檢查同側, 因為網子附近的落點位置有幾 cm
    的誤差, 不足以判斷在哪一側。實例 webm t=97.32: 肉眼確認是觸網後落在擊球方
    (網子右側) 很近處, 但映射成 x=135.4cm (網子左側 1.6cm)。原因是教授的觸桌點估計
    用兩段 y(x) 拋物線求交點, 而觸網後球幾乎沒有水平速度 (每幀 x 只變 1~3px),
    擬合不穩, 估出的觸桌點比實際最低點高約 9px。若位置正確, 下一跳 t=97.72
    (同側 56cm) 本來就會被標; 此容差是在補償這個位置誤差。根本解法是在水平速度小時
    改用 y(t) 擬合或觀察到的最低點估計觸桌點 (待做)。

    為何用 AND 而非 OR: 發球的兩跳時間差 (實測 0.38~0.42s) 和一般回球幾乎一樣短,
    只看時間會把發球第二跳誤刪; 網前短球的兩個落點距離可能很短但分在網子兩側。
    「中間有擊球就不標」是為了漏偵測: 中間漏掉一跳時, 前後兩個真落點也會變成同側。
    比對對象是前一個「偵測到的」落點 (不論是否已被標), 所以連彈三下會一路串起來。
    回傳標記數。
    """
    net_x = TABLE_W_CM / 2
    prev = None
    n = 0
    for ev in sorted(events, key=lambda e: e["t"]):
        if ev.get("type") == "hit":
            prev = None
            continue
        if ev.get("type") != "bounce":
            continue
        ev.pop("net_dribble", None)
        tb, ptb = ev.get("table"), prev.get("table") if prev else None
        if tb and ptb and ((tb[0] < net_x) == (ptb[0] < net_x)
                           or abs(ptb[0] - net_x) <= net_tol) \
                and ev["t"] - prev["t"] <= max_dt \
                and float(np.hypot(tb[0] - ptb[0], tb[1] - ptb[1])) <= max_dist:
            ev["net_dribble"] = True
            n += 1
        prev = ev
    return n


SERVE_GAP = 3.0           # s: 與前一個落點相隔超過此值 → 新回合開始
SERVE_PAIR_DT = 0.6       # s: 發球第 1 跳 → 第 2 跳的最大時間差 (實測 0.38~0.42s)
SERVE_PAIR_DIST = 100.0   # cm: 發球兩跳的最小桌面距離 (實測 153~173cm, 必跨網)


def mark_serves(events: list[dict], gap: float = SERVE_GAP,
                pair_dt: float = SERVE_PAIR_DT, pair_dist: float = SERVE_PAIR_DIST) -> dict:
    """標記發球落點 (ev["serve"] = 1 / 2 / "?"), 不刪除、不改其他欄位。

    回合開始 = 與前一個落點相隔 > gap 秒的落點 (回合間實測 15~36s, 回合內 < 0.9s)。
    只用落點計時, 不用擊球: 死球期間的誤判擊球不該打斷判斷。

    不直接把「回合第一個落點」當發球第 1 跳, 因為第 1 跳常漏抓, 那樣會把接球方的
    第 2 跳誤標成第 1 跳。改看配對: 下一個落點在 pair_dt 內、位於網子另一側、距離
    >= pair_dist、且中間沒有擊球 → 兩跳都抓到, 標 1 和 2。配對不成立 → "?"
    (可能只抓到第 2 跳, 也可能是第 1 跳後軌跡斷了), 不硬判。

    只看「落在桌面上」的落點 (有分區): 地板 logo / 白鞋被誤認成球時, 鏡頭移動會讓它們
    形成桌外假落點; 若參與判斷, 會被當成發球第 1 跳, 或讓真正的回合開始時間錯位。
    回傳各類數量。
    """
    net_x = TABLE_W_CM / 2
    evs = sorted(events, key=lambda e: e["t"])
    for e in evs:
        e.pop("serve", None)
    counts = {1: 0, 2: 0, "?": 0}
    prev_bounce_t = None
    for i, e in enumerate(evs):
        if e.get("type") != "bounce" or not e.get("zone"):
            continue
        is_start = prev_bounce_t is None or e["t"] - prev_bounce_t > gap
        prev_bounce_t = e["t"]
        if not is_start:
            continue
        nxt, hit_between = None, False
        for f in evs[i + 1:]:
            if f.get("type") == "hit":
                hit_between = True
            elif f.get("type") == "bounce" and f.get("zone"):
                nxt = f
                break
        a, b = e["table"], nxt["table"] if nxt else None
        if (b and not hit_between
                and nxt["t"] - e["t"] <= pair_dt
                and (a[0] < net_x) != (b[0] < net_x)
                and float(np.hypot(a[0] - b[0], a[1] - b[1])) >= pair_dist):
            e["serve"], nxt["serve"] = 1, 2
            counts[1] += 1
            counts[2] += 1
        else:
            e["serve"] = "?"
            counts["?"] += 1
    return counts


NET_EDGE_CM = 12.0     # 離網 <= 此距離的落點: 分不出碰網/落桌, 也分不出哪一側
NET_HIT_CM = 25.0      # 離網 <= 此距離的「擊球」: 是球碰網造成的方向反轉, 不是球員擊球


def classify_bounces(events: list[dict], gap: float = SERVE_GAP,
                     net_edge: float = NET_EDGE_CM, net_hit: float = NET_HIT_CM) -> int:
    """以擊球為基準判定哪些落點不算 (ev["uncounted"] = 原因), 不刪除。需先跑 mark_serves。

    規則: 兩次擊球之間, 只有「落在對方那側的第一跳」算落點。
      own_side      落在擊球方自己那側 (掛網沒過、過網後倒旋回來)        → 死球
      second_bounce 對方那側的第二跳以後 (觸網後連彈)                     → 死球
      net_edge      離網 <= net_edge: 碰網或網邊落桌無法區分, 不佔「第一跳」;
                    但這一拍已偵測到碰網時, 網邊落點必是落桌 → 佔第一跳 (仍不算)
      dead_ball     死球後到下一回合開始前的落點 (球員拍球、撿球、球在滾)
      net_dribble   沒偵測到擊球、不知道擊球方時, 退回 mark_net_dribbles 的判斷
    發球第 1 跳例外 (本來就落在發球方), 並以它決定發球方。
    擊球位置在網邊 (<= net_hit) 視為碰網, 不更新擊球方。只考慮桌面上的落點 (有分區)。

    依據 (London 2026): 09:47 / 10:07 / 10:25 三次掛網都是「擊球後第一跳落在擊球方」,
    其中 10:07 掛網後彈到近端線才落下 → 不能用離網距離判斷掛網;
    18:00 碰網那下 (離網 8cm) 被當成落點, 真正的第一跳 (離網 15cm) 反而被當連彈。
    碰網一拍只會發生一次 (Lin vs Moregard 02:32): 02:32.03 打到網帶 (離網附近的桌外事件),
    02:32.47 接球方第一跳被算到網子另一側 0.5cm → 網邊; 若它不佔第一跳, 第二、三跳
    會依序遞補成「第一跳」。碰網的證據: 網邊 (<= net_hit) 的擊球, 或網邊的桌外落點。
    回傳不算的落點數。
    """
    net_x = TABLE_W_CM / 2
    hitter = None          # "L" / "R": 最近一次球員擊球在哪一側
    counted = False        # 這一拍是否已有有效落點
    dead = False
    net_seen = False       # 這一拍是否已偵測到碰網
    prev_t = None
    n = 0
    for e in sorted(events, key=lambda e: e["t"]):
        if e.get("type") == "hit":
            tb = e.get("table")
            e.pop("net_touch", None)
            if not tb:
                continue
            if abs(tb[0] - net_x) <= net_hit:
                e["net_touch"] = True
                net_seen = True
                continue
            if not dead:
                hitter, counted, net_seen = ("L" if tb[0] < net_x else "R"), False, False
            continue
        if e.get("type") != "bounce":
            continue
        e.pop("uncounted", None)
        if not e.get("zone"):
            # 桌外落點另外顯示, 不影響回合狀態; 但在網邊的是球打到網帶 (網帶高於桌面,
            # 透視後映射到桌外), 記為這一拍已碰網
            tb = e.get("table")
            if tb and abs(tb[0] - net_x) <= net_hit:
                net_seen = True
            continue
        if prev_t is None or e["t"] - prev_t > gap:
            hitter, counted, dead, net_seen = None, False, False, False   # 新回合
        prev_t = e["t"]
        side = "L" if e["table"][0] < net_x else "R"
        if e.get("serve") == 1:
            hitter, counted = side, False
            continue
        if dead:
            e["uncounted"] = "dead_ball"
        elif abs(e["table"][0] - net_x) <= net_edge:
            e["uncounted"] = "net_edge"
            if net_seen and hitter is not None:
                if counted:                  # 已有第一跳 → 這是之後的跳
                    dead = True
                counted = True
        elif hitter is None:
            if e.get("net_dribble"):
                e["uncounted"] = "net_dribble"
        elif side == hitter:
            e["uncounted"], dead = "own_side", True
        elif counted:
            e["uncounted"], dead = "second_bounce", True
        else:
            counted = True
        n += "uncounted" in e
    return n


APRON_TOP = -0.05       # 裙板遮罩上緣: 近端邊線上方 5% 桌高 (補角誤差餘裕)
APRON_BOTTOM = 0.35     # 裙板遮罩下緣: 近端邊線下方 35% 桌高
APRON_MAX_CONF = 0.6    # 只擋低於此信心的偵測


def in_apron(quad: np.ndarray, x: float, y: float, conf: float) -> bool:
    """球偵測是否落在球桌前裙板 (近端邊線下方、桌子左右範圍內) 且信心低。

    London 2026 的裙板上有白色 DHS logo, 球模型以 0.31~0.59 的信心把它認成球,
    插進真實軌跡形成假落點 (t=54.38/56.48/59.40)。真球出現在這個區域時信心都 >=0.85,
    且多半在桌子左右範圍之外 (飛出桌端)。

    上緣放在邊線「上方」5%, 因為發球前桌角被擋、補角偏差時, logo 會被算在邊線上方
    3~4%。只靠位置會與擦邊球 (球心約在邊線上方 1.3%) 重疊, 所以加上信心條件。
    距離以桌面在畫面中的高度正規化, zoom 時不必調整。
    """
    if conf >= APRON_MAX_CONF:
        return False
    FL, FR, NR, NL = quad
    edge = NR - NL
    length = float(np.linalg.norm(edge))
    u = edge / max(length, 1e-9)
    n = np.array([-u[1], u[0]])
    if n[1] < 0:            # 法向量取朝畫面下方
        n = -n
    table_h = (np.linalg.norm(NL - FL) + np.linalg.norm(NR - FR)) / 2
    v = np.array([x, y]) - NL
    below = float(v @ n) / max(table_h, 1e-9)
    along = float(v @ u) / max(length, 1e-9)
    return 0.0 <= along <= 1.0 and APRON_TOP <= below <= APRON_BOTTOM


def ball_detections(ball: BallModel, frame, conf_min: float) -> list[dict]:
    """跑 Core ML 球偵測器,回傳原圖座標的候選點。"""
    from PIL import Image
    h, w = frame.shape[:2]
    s = ball.imgsz
    rgb = cv2.cvtColor(cv2.resize(frame, (s, s)), cv2.COLOR_BGR2RGB)
    raw = ball.ml.predict({"image": Image.fromarray(rgb)})
    out = []
    if ball.arch == "rfdetr":
        boxes = np.asarray(raw["pred_boxes"]).reshape(-1, 4)
        logits = np.asarray(raw["pred_logits"])
        scores = (1 / (1 + np.exp(-logits.reshape(boxes.shape[0], -1)))).max(axis=1)
        for (cx, cy, _bw, _bh), cf in zip(boxes[scores >= conf_min], scores[scores >= conf_min]):
            out.append({"x": float(cx * w), "y": float(cy * h), "conf": float(cf)})
        return out
    arr = np.asarray(next(iter(raw.values()))).reshape(-1, 6)
    for x1, y1, x2, y2, cf, _cls in arr[arr[:, 4] >= conf_min]:
        out.append({"x": float((x1 + x2) / 2 * w / s),
                    "y": float((y1 + y2) / 2 * h / s),
                    "conf": float(cf)})
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--prof-repo", default=str(DEFAULT_PROF_REPO), help="PingPongTracker repo 路徑")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--table-backend", choices=("coreml", "torch"), default="coreml",
                    help="桌面模型執行方式: coreml (預設, Models/TableDetector.mlpackage) / torch (checkpoint)")
    ap.add_argument("--ball-model", default=None,
                    help="球偵測 .mlpackage (預設: Models/BallDetector_rfdetr_20260930.mlpackage, "
                         "RF-DETR Large)。YOLO / RF-DETR 格式自動判斷")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--conf", type=float, default=0.3, help="球偵測信心門檻")
    ap.add_argument("--zone-expand", type=float, default=0.4 * TABLE_W_CM,
                    help="場外閘門寬容度 (cm,超出桌面此距離的球偵測視為誤判)。"
                         "預設 0.4x 桌長 = 110cm;教授原值 700 是舊的 500 單位空間,"
                         "對轉播畫面幾乎不過濾")
    ap.add_argument("--min-y-speed-frac", type=float, default=0.12,
                    help="落點 y 速度噪音底限,以「桌面在畫面中的高度」的倍率表示 (每秒)。"
                         "教授原值 40px/s 是針對桌子佔滿畫面的取景;轉播畫面桌高僅約 130px,"
                         "同樣的物理反彈像素速度小 4~5 倍,固定 px 門檻會誤殺真實落點")
    ap.add_argument("--min-y-speed", type=float, default=None,
                    help="直接指定 y 速度底限 (px/s),覆蓋 --min-y-speed-frac")
    ap.add_argument("--dribble-max-dt", type=float, default=DRIBBLE_MAX_DT,
                    help="觸網連續彈跳: 與前一落點的最大時間差 (s)")
    ap.add_argument("--dribble-max-dist", type=float, default=DRIBBLE_MAX_DIST,
                    help="觸網連續彈跳: 與前一落點的最大桌面距離 (cm)")
    ap.add_argument("--dribble-net-tol", type=float, default=DRIBBLE_NET_TOL,
                    help="觸網連續彈跳: 前一跳離網子 <= 此距離 (cm) 時不檢查同側")
    ap.add_argument("--no-apron-mask", action="store_true",
                    help="停用球桌前裙板遮罩 (供對照)")
    ap.add_argument("--no-fix-debounce", action="store_true",
                    help="停用去抖動修正 (保留教授原行為, 供對照)")
    ap.add_argument("--no-touch-refine", action="store_true",
                    help="停用觸桌點重估, 保留教授的 y(x) 拋物線交點 (供對照)")
    ap.add_argument("--no-kink", action="store_true",
                    help="停用折角落點補抓 (y 速度突然往上改變, 但沒有由正轉負)")
    ap.add_argument("--out", default=None, help="輸出 JSON (預設 data/landing_<影片名>.json)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="重跑球偵測與桌面模型 (第一階段), 覆蓋 data/cache/ 的快取")
    return ap


CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def cache_path(args) -> Path:
    """第一階段快取檔名: 影片 + 片段 + 球偵測門檻 + 模型。換任何一個都會重跑。"""
    ball = Path(args.ball_model).stem if args.ball_model else DEFAULT_BALL_MODEL.stem
    return CACHE_DIR / (f"{Path(args.video).stem}_s{args.start:g}_d{args.dur:g}"
                        f"_c{args.conf:g}_{ball}_{args.table_backend}.npz")


def run_detectors(args) -> dict:
    """第一階段 (慢): 逐幀跑桌面模型與球偵測。

    存桌面模型的「原始輸出」(4 角、分數、presence), 不存 TableTracker 的結果,
    這樣改追蹤器 (例如桌面鎖定) 時快取仍可用。
    """
    ball_model = BallModel(args.ball_model)
    print(ball_model)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    table_model = load_table_model(args.ckpt, device, args.table_backend)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check_aspect(src_w, src_h)
    start_f = int(args.start * fps)
    n_frames = int(args.dur * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    fis, coords_l, scores_l, presence_l, dets_l = [], [], [], [], []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        # 一律在 1280x720 處理 (門檻都在此尺寸校準); JSON 內所有像素座標都是處理座標
        frame = to_proc(frame)
        coords, scores, presence = predict(table_model, frame, device, PROC_W, PROC_H)
        fis.append(start_f + i)
        coords_l.append(coords)
        scores_l.append(scores)
        presence_l.append(presence)
        dets_l.append(ball_detections(ball_model, frame, args.conf))
    cap.release()
    return {"fps": fps, "src_size": [src_w, src_h], "frame": np.array(fis),
            "coords": np.array(coords_l, np.float64).reshape(-1, 4, 2),
            "scores": np.array(scores_l, np.float64).reshape(-1, 4),
            "presence": np.array(presence_l, np.float64), "dets": dets_l}


def load_or_run_detectors(args) -> dict:
    path = cache_path(args)
    if path.exists() and not args.refresh_cache:
        z = np.load(path, allow_pickle=False)
        print(f"讀取快取 {path.relative_to(PROJECT_ROOT)}")
        return {"fps": float(z["fps"]), "src_size": z["src_size"].tolist(), "frame": z["frame"],
                "coords": z["coords"], "scores": z["scores"], "presence": z["presence"],
                "dets": json.loads(str(z["dets_json"]))}
    raw = run_detectors(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fps=raw["fps"], src_size=np.array(raw["src_size"]),
                        frame=raw["frame"], coords=raw["coords"], scores=raw["scores"],
                        presence=raw["presence"], dets_json=json.dumps(raw["dets"]))
    print(f"寫入快取 {path.relative_to(PROJECT_ROOT)}")
    return raw


def analyze(raw: dict, args, de) -> tuple[dict, dict]:
    """第二階段 (快): 桌面追蹤 → 球偵測過濾 → 事件 → 落點規則。回傳 (輸出 JSON, 統計)。"""
    fps = raw["fps"]
    w, h = PROC_W, PROC_H
    tracker = TableTracker(w, h)

    frames: list[dict] = []
    H_by_frame: dict[int, np.ndarray] = {}
    quads_px: list[np.ndarray] = []   # 供 MIN_Y_SPEED 依桌面畫面大小縮放
    n_table = 0
    quad_by_frame: dict[int, np.ndarray] = {}   # 供裙板遮罩
    for k, fi in enumerate(raw["frame"].tolist()):
        quad = tracker.update(raw["coords"][k], raw["scores"][k], float(raw["presence"][k]))
        if quad is not None:
            # 我們的 [FL,FR,NR,NL] → 他的 [NL,NR,FR,FL] (互為反序)
            H_by_frame[fi] = de.build_homography(quad[::-1].astype(np.float32))
            quads_px.append(quad)
            quad_by_frame[fi] = quad
            n_table += 1
        # dets 會被下面的遮罩/閘門改寫, 複製一份以免動到快取內容
        frames.append({"t": fi / fps, "frame": fi, "dets": [dict(d) for d in raw["dets"][k]]})

    # 球桌前裙板遮罩。全部幀處理完再做, 才能往前「和往後」找最近的桌面框:
    # 追蹤器要連續幾幀一致才確認桌面, 確認前那幾幀 (常是發球前) 沒有框, 只往前找會漏。
    n_apron = 0
    if not args.no_apron_mask and quad_by_frame:
        qfs = np.array(sorted(quad_by_frame))
        for fr in frames:
            if not fr["dets"]:
                continue
            k = int(qfs[np.argmin(np.abs(qfs - fr["frame"]))])
            if abs(k - fr["frame"]) > 10:
                continue
            kept = [d for d in fr["dets"]
                    if not in_apron(quad_by_frame[k], d["x"], d["y"], d["conf"])]
            n_apron += len(fr["dets"]) - len(kept)
            fr["dets"] = kept

    # 逐幀桌面區域閘門 (取代他 build_track 內用固定 H 的那段)
    expand = args.zone_expand
    sorted_fis = sorted(H_by_frame)

    def H_near(fi: int, max_dt: float = 0.5):
        """取最接近的可用 H (桌面偵測短暫中斷時仍能過濾);超出時間窗回 None。"""
        if not sorted_fis:
            return None
        best = min(sorted_fis, key=lambda k: abs(k - fi))
        return H_by_frame[best] if abs(best - fi) / fps <= max_dt else None

    for fr in frames:
        H = H_by_frame.get(fr["frame"])
        if H is None:
            H = H_near(fr["frame"])
        if H is None:
            # 無任何可用桌面資訊 → 無法驗證球位置,且該處落點也無法映射,直接丟棄
            fr["dets"] = []
            continue
        kept = []
        for d in fr["dets"]:
            tx, ty = de.map_point(H, d["x"], d["y"])
            if -expand <= tx <= de.TABLE_W + expand and -expand <= ty <= de.TABLE_H + expand:
                kept.append(d)
        fr["dets"] = kept

    # y 速度底限依桌面在畫面中的大小縮放 (見 --min-y-speed-frac 說明)
    table_px_h = (float(np.median([q[:, 1].max() - q[:, 1].min() for q in quads_px]))
                  if quads_px else None)
    if args.min_y_speed is not None:
        de.MIN_Y_SPEED = args.min_y_speed
    elif table_px_h is not None:
        de.MIN_Y_SPEED = args.min_y_speed_frac * table_px_h
    print(f"MIN_Y_SPEED = {de.MIN_Y_SPEED:.1f} px/s")

    # 教授的演算法 (H=None: 映射改在外部用逐幀 H 做)
    track = de.build_track(frames, None, args.conf)
    # 他的 detect_events/split_runs 假設 track 非空,空軌跡會 IndexError
    events = [] if not track else de.detect_events(track, None)[0]
    n_kink = (0 if args.no_kink or table_px_h is None
              else detect_kink_bounces(track, events, table_px_h))
    n_touch = 0 if args.no_touch_refine else refine_touch_points(events)

    def nearest_H(t: float):
        if not H_by_frame:
            return None
        fi = min(H_by_frame, key=lambda k: abs(k / fps - t))
        return H_by_frame[fi] if abs(fi / fps - t) <= 0.2 else None

    n_mapped = 0
    for ev in events:
        # 擊球也映射: classify_bounces 需要知道擊球方在網子哪一側。
        # 擊球點在桌面上方 (不在桌面平面), 映射座標只用來分左右, 不代表實際位置。
        H = nearest_H(ev["t"])
        if H is None:
            continue
        tx, ty = de.map_point(H, ev["x"], ev["y"])
        ev["table"] = [round(tx, 1), round(ty, 1)]
        if ev.get("type") == "bounce":
            ev["zone"] = de.zone_of(tx, ty)
            n_mapped += 1
            if "xy_prof" in ev:   # 供對照: 教授的觸桌點估計映射到哪
                px, py = de.map_point(H, ev["xy_prof"][0], ev["xy_prof"][1])
                ev["table_prof"] = [round(px, 1), round(py, 1)]
    n_edge = apply_edge_buffer(events)
    n_dribble = mark_net_dribbles(events, args.dribble_max_dt, args.dribble_max_dist,
                                  args.dribble_net_tol)
    n_serve = mark_serves(events)
    n_uncounted = classify_bounces(events)

    result = {"video": args.video, "start": args.start, "dur": args.dur, "fps": fps,
              "src_size": raw["src_size"], "proc_size": [PROC_W, PROC_H],
              "zone_expand": expand, "conf": args.conf,
              "frames_with_table": n_table, "frames_total": len(frames),
              "frames_with_ball_det": sum(1 for f in frames if f["dets"]),
              "apron_masked": n_apron,
              "track_points": len(track), "track": track, "events": events}
    stats = {"n_table": n_table, "n_frames": len(frames), "n_track": len(track), "n_apron": n_apron,
             "n_mapped": n_mapped, "n_dribble": n_dribble, "n_serve": n_serve,
             "n_uncounted": n_uncounted, "n_edge": n_edge, "n_touch": n_touch, "n_kink": n_kink}
    return result, stats


def main() -> None:
    args = build_parser().parse_args()
    de = load_prof_module(Path(args.prof_repo))
    if not args.no_fix_debounce:
        patch_debounce(de)
    raw = load_or_run_detectors(args)
    result, st = analyze(raw, args, de)
    events = result["events"]

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"landing_{Path(args.video).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    bounces = [e for e in events if e.get("type") == "bounce"]
    hits = [e for e in events if e.get("type") == "hit"]
    n_table, n_frames = st["n_table"], st["n_frames"]
    print(f"影格 {n_frames}  有桌面 {n_table} ({n_table/max(n_frames,1):.0%})  "
          f"球軌跡點 {st['n_track']}  裙板遮罩擋掉 {st['n_apron']} 個偵測")
    print(f"事件: 擊球 {len(hits)}, 落點 {len(bounces)} (其中 {st['n_mapped']} 個成功映射到桌面座標, "
          f"{st['n_dribble']} 個標為觸網連續彈跳)")
    print(f"發球: 兩跳都抓到 {st['n_serve'][1]} 次, 只抓到一跳/無法確定 {st['n_serve']['?']} 次")
    print(f"不算的落點 (掛網/過網彈回/連彈/網邊/死球): {st['n_uncounted']} 個")
    print(f"擦邊緩衝 ({EDGE_BUFFER_CM:.0f}cm) 拉回桌上: {st['n_edge']} 個")
    print(f"觸桌點重估 (y(t) 交點): {st['n_touch']}/{len(bounces)} 個落點")
    print(f"折角落點補抓: {st['n_kink']} 個" + "".join(
        f"  t={e['t']:.2f}s ({e['kink_dv']:.0f}px/s)" for e in bounces if e.get("kink")))
    for e in bounces[:12]:
        z = e.get("zone")
        tb = e.get("table")
        print(f"  t={e['t']:.2f}s  影像({e['x']:.0f},{e['y']:.0f})  "
              f"桌面{tb}  分區 {z}" + ("  [觸網彈跳]" if e.get("net_dribble") else "")
              + (f"  [發球 {e['serve']}]" if e.get("serve") else "")
              + (f"  [不算: {e['uncounted']}]" if e.get("uncounted") else ""))
    print(f"\n輸出 {out_path}")




if __name__ == "__main__":
    main()
