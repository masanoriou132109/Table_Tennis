#!/usr/bin/env python3
"""端到端落點分析: 本專案的逐幀桌角 + PingPongTracker 的球偵測與事件演算法。

串接方式 (完全不修改教授的程式碼):
  1. 本專案 TableKeypointNet + TableTracker → 每幀桌面 4 角 → 每幀 homography
  2. 教授的 Models/BallDetector.mlpackage (YOLO26, Core ML) → 每幀球偵測
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

from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

BALL_INPUT = 640

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


def ball_detections(ml, frame, conf_min: float) -> list[dict]:
    """跑 Core ML 球偵測器,回傳原圖座標的候選點。"""
    from PIL import Image
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame, (BALL_INPUT, BALL_INPUT)), cv2.COLOR_BGR2RGB)
    raw = ml.predict({"image": Image.fromarray(rgb)})
    arr = np.asarray(next(iter(raw.values()))).reshape(-1, 6)
    out = []
    for x1, y1, x2, y2, cf, _cls in arr[arr[:, 4] >= conf_min]:
        out.append({"x": float((x1 + x2) / 2 * w / BALL_INPUT),
                    "y": float((y1 + y2) / 2 * h / BALL_INPUT),
                    "conf": float(cf)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--prof-repo", default=str(PROJECT_ROOT / "external" / "PingPongTracker"), help="PingPongTracker repo 路徑")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--conf", type=float, default=0.3, help="球偵測信心門檻")
    ap.add_argument("--zone-expand", type=float, default=0.4 * TABLE_W_CM,
                    help="場外閘門寬容度 (cm,超出桌面此距離的球偵測視為誤判)。"
                         "預設 0.4x 桌長 = 110cm;教授原值 700 是舊的 500 單位空間,"
                         "對轉播畫面幾乎不過濾")
    ap.add_argument("--out", default=None, help="輸出 JSON (預設 data/landing_<影片名>.json)")
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
    tracker = TableTracker(w, h)
    start_f = int(args.start * fps)
    n_frames = int(args.dur * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frames: list[dict] = []
    H_by_frame: dict[int, np.ndarray] = {}
    n_table = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        fi = start_f + i
        t = fi / fps

        coords, scores, presence = predict(table_model, frame, device, w, h)
        quad = tracker.update(coords, scores, presence)
        if quad is not None:
            # 我們的 [FL,FR,NR,NL] → 他的 [NL,NR,FR,FL] (互為反序)
            H_by_frame[fi] = de.build_homography(quad[::-1].astype(np.float32))
            n_table += 1

        frames.append({"t": t, "frame": fi,
                       "dets": ball_detections(ball_model, frame, args.conf)})
    cap.release()

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

    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"landing_{Path(args.video).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"video": args.video, "start": args.start, "dur": args.dur, "fps": fps,
         "zone_expand": expand, "conf": args.conf,
         "frames_with_table": n_table, "frames_total": len(frames),
         "frames_with_ball_det": sum(1 for f in frames if f["dets"]),
         "track_points": len(track), "track": track, "events": events},
        ensure_ascii=False, indent=2))

    bounces = [e for e in events if e.get("type") == "bounce"]
    hits = [e for e in events if e.get("type") == "hit"]
    print(f"影格 {len(frames)}  有桌面 {n_table} ({n_table/max(len(frames),1):.0%})  "
          f"球軌跡點 {len(track)}")
    print(f"事件: 擊球 {len(hits)}, 落點 {len(bounces)} (其中 {n_mapped} 個成功映射到桌面座標)")
    for e in bounces[:12]:
        z = e.get("zone")
        tb = e.get("table")
        print(f"  t={e['t']:.2f}s  影像({e['x']:.0f},{e['y']:.0f})  "
              f"桌面{tb}  分區 {z}")
    print(f"\n輸出 {out_path}")


if __name__ == "__main__":
    main()
