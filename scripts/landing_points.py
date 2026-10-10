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

from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from src.video_io import PROC_H, PROC_W, check_aspect, to_proc  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

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
    回傳各類數量。
    """
    net_x = TABLE_W_CM / 2
    evs = sorted(events, key=lambda e: e["t"])
    for e in evs:
        e.pop("serve", None)
    counts = {1: 0, 2: 0, "?": 0}
    prev_bounce_t = None
    for i, e in enumerate(evs):
        if e.get("type") != "bounce":
            continue
        is_start = prev_bounce_t is None or e["t"] - prev_bounce_t > gap
        prev_bounce_t = e["t"]
        if not is_start:
            continue
        nxt, hit_between = None, False
        for f in evs[i + 1:]:
            if f.get("type") == "hit":
                hit_between = True
            elif f.get("type") == "bounce":
                nxt = f
                break
        a, b = e.get("table"), nxt.get("table") if nxt else None
        if (a and b and not hit_between
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--prof-repo", default=str(DEFAULT_PROF_REPO), help="PingPongTracker repo 路徑")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
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
    ap.add_argument("--out", default=None, help="輸出 JSON (預設 data/landing_<影片名>.json)")
    args = ap.parse_args()

    repo = Path(args.prof_repo)
    de = load_prof_module(repo)
    if not args.no_fix_debounce:
        patch_debounce(de)
    ball_model = BallModel(args.ball_model)
    print(ball_model)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    table_model = TableKeypointNet(pretrained=False).to(device)
    table_model.load_state_dict(ckpt["model"])
    table_model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    check_aspect(src_w, src_h)
    # 一律在 1280x720 處理 (門檻都在此尺寸校準); JSON 內所有像素座標都是處理座標
    w, h = PROC_W, PROC_H
    tracker = TableTracker(w, h)
    start_f = int(args.start * fps)
    n_frames = int(args.dur * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frames: list[dict] = []
    H_by_frame: dict[int, np.ndarray] = {}
    quads_px: list[np.ndarray] = []   # 供 MIN_Y_SPEED 依桌面畫面大小縮放
    n_table = 0
    quad_by_frame: dict[int, np.ndarray] = {}   # 供裙板遮罩
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = to_proc(frame)
        fi = start_f + i
        t = fi / fps

        coords, scores, presence = predict(table_model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        if quad is not None:
            # 我們的 [FL,FR,NR,NL] → 他的 [NL,NR,FR,FL] (互為反序)
            H_by_frame[fi] = de.build_homography(quad[::-1].astype(np.float32))
            quads_px.append(quad)
            n_table += 1

        if quad is not None:
            quad_by_frame[fi] = quad
        frames.append({"t": t, "frame": fi,
                       "dets": ball_detections(ball_model, frame, args.conf)})
    cap.release()

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
    if args.min_y_speed is not None:
        de.MIN_Y_SPEED = args.min_y_speed
    elif quads_px:
        table_px_h = float(np.median([q[:, 1].max() - q[:, 1].min() for q in quads_px]))
        de.MIN_Y_SPEED = args.min_y_speed_frac * table_px_h
    print(f"MIN_Y_SPEED = {de.MIN_Y_SPEED:.1f} px/s")

    # 教授的演算法 (H=None: 映射改在外部用逐幀 H 做)
    track = de.build_track(frames, None, args.conf)
    # 他的 detect_events/split_runs 假設 track 非空,空軌跡會 IndexError
    events = [] if not track else de.detect_events(track, None)[0]

    def nearest_H(t: float):
        if not H_by_frame:
            return None
        fi = min(H_by_frame, key=lambda k: abs(k / fps - t))
        return H_by_frame[fi] if abs(fi / fps - t) <= 0.2 else None

    n_mapped = 0
    for ev in events:
        if ev.get("type") != "bounce":
            continue
        H = nearest_H(ev["t"])
        if H is None:
            continue
        tx, ty = de.map_point(H, ev["x"], ev["y"])
        ev["table"] = [round(tx, 1), round(ty, 1)]
        ev["zone"] = de.zone_of(tx, ty)
        n_mapped += 1
    n_dribble = mark_net_dribbles(events, args.dribble_max_dt, args.dribble_max_dist,
                                  args.dribble_net_tol)
    n_serve = mark_serves(events)

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"landing_{Path(args.video).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"video": args.video, "start": args.start, "dur": args.dur, "fps": fps,
         "src_size": [src_w, src_h], "proc_size": [PROC_W, PROC_H],
         "zone_expand": expand, "conf": args.conf,
         "frames_with_table": n_table, "frames_total": len(frames),
         "frames_with_ball_det": sum(1 for f in frames if f["dets"]),
         "apron_masked": n_apron,
         "track_points": len(track), "track": track, "events": events},
        ensure_ascii=False, indent=2))

    bounces = [e for e in events if e.get("type") == "bounce"]
    hits = [e for e in events if e.get("type") == "hit"]
    print(f"影格 {len(frames)}  有桌面 {n_table} ({n_table/max(len(frames),1):.0%})  "
          f"球軌跡點 {len(track)}  裙板遮罩擋掉 {n_apron} 個偵測")
    print(f"事件: 擊球 {len(hits)}, 落點 {len(bounces)} (其中 {n_mapped} 個成功映射到桌面座標, "
          f"{n_dribble} 個標為觸網連續彈跳)")
    print(f"發球: 兩跳都抓到 {n_serve[1]} 次, 只抓到一跳/無法確定 {n_serve['?']} 次")
    for e in bounces[:12]:
        z = e.get("zone")
        tb = e.get("table")
        print(f"  t={e['t']:.2f}s  影像({e['x']:.0f},{e['y']:.0f})  "
              f"桌面{tb}  分區 {z}" + ("  [觸網彈跳]" if e.get("net_dribble") else "")
              + (f"  [發球 {e['serve']}]" if e.get("serve") else ""))
    print(f"\n輸出 {out_path}")


if __name__ == "__main__":
    main()
