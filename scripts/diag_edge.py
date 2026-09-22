#!/usr/bin/env python3
"""診斷: edge 校正到底抓到了什麼邊,以及 RANSAC 選了哪些點。

對每條邊產生一張「攤平的邊帶」(edge strip):沿邊方向為橫軸、沿法線方向為縱軸。
真正的桌面邊界在這張圖裡會是一條接近水平的亮暗分界。疊上:
  紅點  = 逐欄偵測到的梯度峰 (未過濾)
  黃線  = 先驗邊線位置 (法線偏移 0)
  綠線  = RANSAC 擬合並被接受的邊線 → 若綠線貼著真實分界, 就是抓對了
  綠色刻度 (底部) = RANSAC 內點涵蓋的跨距

另輸出全幀圖: 綠框=先驗, 橘框=精修後, 綠點=內點, 紅點=被丟棄的點。

用法:
    python scripts/diag_edge.py VIDEO --t 95.1 [--out-dir data/diag_edge]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src import edge_refine as ER  # noqa: E402
from src.model import TableKeypointNet  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

EDGE_NAMES = ["far(FL-FR)", "right(FR-NR)", "near(NR-NL)", "left(NL-FL)"]
CN = ["FL", "FR", "NR", "NL"]
ZOOM = 5          # 邊帶縱向放大
ALONG = 240       # 邊帶橫向取樣數


def edge_strip(gray, p0, p1):
    """攤平邊帶 + 逐欄梯度峰位置 (法線偏移, None 表示該欄沒找到)。"""
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 20:
        return None, None, 0.0
    d = d / L
    n = np.array([-d[1], d[0]])
    offs = np.arange(-ER.SEARCH, ER.SEARCH + 1, dtype=np.float64)
    strip = np.zeros((len(offs), ALONG), np.float32)
    peaks: list[float | None] = []
    h, w = gray.shape
    for j, t in enumerate(np.linspace(0.0, 1.0, ALONG)):
        base = p0 + d * (t * L)
        pts = base[None, :] + n[None, :] * offs[:, None]
        xs = pts[:, 0].astype(np.float32).reshape(-1, 1)
        ys = pts[:, 1].astype(np.float32).reshape(-1, 1)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            peaks.append(None)
            continue
        prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR).ravel()
        strip[:, j] = prof
        g = np.abs(np.diff(prof))
        k = int(np.argmax(g))
        peaks.append(offs[k] + 0.5 if g[k] >= ER.MIN_GRAD else None)
    return strip, peaks, L


def line_offset_at(line, p0, d, n, t, L):
    """擬合線在參數 t 處相對先驗邊線的法線偏移 (畫在邊帶上用)。"""
    base = p0 + d * (t * L)
    a, b, c = line
    den = a * n[0] + b * n[1]
    if abs(den) < 1e-9:
        return None
    return -(a * base[0] + b * base[1] + c) / den


def render_strip(strip, peaks, info, p0, p1, title):
    s = cv2.normalize(strip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    H = s.shape[0]
    img = cv2.cvtColor(cv2.resize(s, (ALONG * 2, H * ZOOM), interpolation=cv2.INTER_NEAREST),
                       cv2.COLOR_GRAY2BGR)

    def y_of(off):
        return int((off + ER.SEARCH) * ZOOM + ZOOM / 2)

    cv2.line(img, (0, y_of(0)), (img.shape[1], y_of(0)), (0, 200, 200), 1)   # 先驗位置
    for j, off in enumerate(peaks):
        if off is not None:
            cv2.circle(img, (j * 2, y_of(off)), 2, (60, 60, 255), -1)        # 原始峰

    if info["line"] is not None:
        d = p1 - p0
        L = float(np.linalg.norm(d))
        d = d / L
        n = np.array([-d[1], d[0]])
        prev = None
        for j, t in enumerate(np.linspace(0.0, 1.0, ALONG)):
            off = line_offset_at(info["line"], p0, d, n, t, L)
            if off is None or abs(off) > ER.SEARCH:
                prev = None
                continue
            cur = (j * 2, y_of(off))
            if prev is not None:
                cv2.line(img, prev, cur, (0, 255, 0), 2)                     # 擬合線
            prev = cur
        # 底部標出內點跨距
        ts = np.linspace(0.0, 1.0, ALONG)
        frac = info["span"] / max(info["length"], 1e-6)
        cx = info["centroid"]
        if cx is not None:
            t_mid = float(np.clip(np.dot(cx - p0, d) / L, 0, 1))
            j0 = int(np.clip((t_mid - frac / 2) * ALONG, 0, ALONG - 1)) * 2
            j1 = int(np.clip((t_mid + frac / 2) * ALONG, 0, ALONG - 1)) * 2
            cv2.line(img, (j0, img.shape[0] - 4), (j1, img.shape[0] - 4), (0, 255, 0), 3)

    bar = np.full((24, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--warmup", type=float, default=6.0, help="先跑幾秒建立追蹤器先驗")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "diag_edge"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = TableTracker(w, h)
    start = max(0, int((args.t - args.warmup) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    quad = conf = frame = ref = None
    target = int(args.t * fps)
    for fi in range(start, target + 1):
        ok, fr = cap.read()
        if not ok:
            break
        c, s, p = predict(model, fr, device, w, h)
        q = tracker.update(c, s, p)
        if fi == target:
            quad, conf, frame = q, tracker.smoother.confident.copy(), fr.copy()
            ref = (tracker.smoother.reference.copy()
                   if tracker.smoother.reference is not None else None)
    cap.release()
    if quad is None:
        print(f"t={args.t}s 追蹤器未輸出四邊形")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.video).stem[:20]}_t{args.t:.2f}"
    print(f"t={args.t}s  可信角={[CN[i] for i in range(4) if conf[i]]}  "
          f"被遮={[CN[i] for i in range(4) if not conf[i]]}")

    refined, ok_mask, infos = ER.refine_quad(frame, quad, conf, reference=ref)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
                            (0, 0), 1.2)

    strips = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        info = infos[i]
        strip, peaks, L = edge_strip(gray, quad[a].astype(np.float64), quad[b].astype(np.float64))
        if strip is None:
            continue
        acc = info.get("accepted", False)
        title = (f"{EDGE_NAMES[i]} {CN[a]}{'' if conf[a] else '(occ)'}->"
                 f"{CN[b]}{'' if conf[b] else '(occ)'}  "
                 f"peaks {len(info['all_pts'])} -> polarity {len(info['kept_pts'])} -> "
                 f"inliers {info['n_inliers']}   span {info['span']:.0f}/{info['length']:.0f}px"
                 f"   angle dev {info.get('angle_dev', float('nan')):.1f}deg"
                 f"   {'ACCEPTED' if acc else 'REJECTED'}")
        print("  " + title)
        strips.append(render_strip(strip, peaks, info,
                                   quad[a].astype(np.float64), quad[b].astype(np.float64), title))

    if strips:
        wmax = max(s.shape[1] for s in strips)
        strips = [cv2.copyMakeBorder(s, 0, 10, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT,
                                     value=(30, 30, 30)) for s in strips]
        cv2.imwrite(str(out / f"{tag}_strips.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 95])

    vis = frame.copy()
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(vis, [refined.astype(np.int32)], True, (0, 180, 255), 2)
    for i in range(4):
        info = infos[i]
        inl = info["inlier_mask"]
        for k, p in enumerate(info["kept_pts"]):
            green = inl is not None and inl[k]
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3,
                       (0, 255, 0) if green else (60, 60, 255), -1)
    for i in range(4):
        cv2.circle(vis, tuple(quad[i].astype(int)), 7,
                   (0, 255, 0) if conf[i] else (0, 0, 255), -1)
        if ok_mask[i]:
            cv2.circle(vis, tuple(refined[i].astype(int)), 9, (0, 180, 255), 2)
    moved = [f"{CN[i]}{np.linalg.norm(refined[i]-quad[i]):.0f}px"
             f"{'(proj)' if ok_mask[i] == 2 else ''}" for i in range(4) if ok_mask[i]]
    cv2.putText(vis, f"green=prior/inliers  orange=refined  red=rejected pts   moved: {moved}",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{tag}_frame.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  精修的角: {moved if moved else '無'}")
    print(f"輸出於 {out}/{tag}_strips.jpg 與 _frame.jpg")


if __name__ == "__main__":
    main()
