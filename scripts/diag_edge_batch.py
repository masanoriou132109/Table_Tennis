#!/usr/bin/env python3
"""跨場地批次產生 edge 校正診斷圖,供人工檢查。

每部影片挑三類代表性影格 (這些是最能暴露問題的案例):
  - move  : 精修位移最大的 2 幀  → 若抓錯邊, 這裡最明顯
  - reject: 有邊被拒絕的 1 幀    → 確認失敗時是安全地不動
  - clean : 四角全可見的 1 幀    → 確認沒把本來就對的框弄壞

輸出 <out-dir>/<venue>_<kind>_t<秒>_frame.jpg 與 _strips.jpg。

用法:
    python scripts/diag_edge_batch.py [--dur 60]
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
from scripts.diag_edge import CN, EDGE_NAMES, edge_strip, render_strip  # noqa: E402
from scripts.infer_video import predict  # noqa: E402

VIDEOS = [
    ("Doha", "videos/20250523World Table Tennis Championships-Wang Chuqin.mp4", 200),
    ("Vegas", "videos/20250712USA-Tomokazu Harimoto.mp4", 100),
    ("Chongqing", "videos/20250312Chongqing-Jang Woojin.mp4", 116),
    ("IncheonWebm", "videos/2025WTTInche-Lee Sang Su.webm", 60),
    ("IncheonSora", "videos/Incheon Sora Matsushima 2025.mp4", 120),
]


def render_case(frame, quad, conf, ref, tag, out: Path):
    refined, ok_mask, infos = ER.refine_quad(frame, quad, conf, reference=ref)
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
                            (0, 0), 1.2)
    strips = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        info = infos[i]
        strip, peaks, _ = edge_strip(gray, quad[a].astype(np.float64), quad[b].astype(np.float64))
        if strip is None:
            continue
        title = (f"{EDGE_NAMES[i]} {CN[a]}{'' if conf[a] else '(occ)'}->"
                 f"{CN[b]}{'' if conf[b] else '(occ)'}  "
                 f"peaks {len(info['all_pts'])}->pol {len(info['kept_pts'])}->inl {info['n_inliers']}"
                 f"  span {info['span']:.0f}/{info['length']:.0f}px"
                 f"  ang {info.get('angle_dev', float('nan')):.1f}deg"
                 f"  {'ACCEPTED' if info.get('accepted') else 'REJECTED'}")
        print("      " + title)
        strips.append(render_strip(strip, peaks, info, quad[a].astype(np.float64),
                                   quad[b].astype(np.float64), title))
    if strips:
        wmax = max(s.shape[1] for s in strips)
        strips = [cv2.copyMakeBorder(s, 0, 10, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT,
                                     value=(30, 30, 30)) for s in strips]
        cv2.imwrite(str(out / f"{tag}_strips.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 92])

    vis = frame.copy()
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(vis, [refined.astype(np.int32)], True, (0, 180, 255), 2)
    for info in infos:
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
    moved = [f"{CN[i]}{np.linalg.norm(refined[i]-quad[i]):.0f}" for i in range(4) if ok_mask[i]]
    cv2.putText(vis, f"{tag}   green=prior/inliers  orange=refined  red=rejected pts",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(vis, f"moved: {moved if moved else 'none'}", (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
    cv2.imwrite(str(out / f"{tag}_frame.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dur", type=float, default=60.0, help="每部影片掃描秒數")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "diag_edge"))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for venue, path, start_s in VIDEOS:
        if not Path(path).exists():
            print(f"[跳過] 找不到 {path}")
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        tracker = TableTracker(w, h)
        moves, rejects, cleans = [], [], []
        for i in range(int(args.dur * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            t = start_s + i / fps
            c, s, p = predict(model, fr, device, w, h)
            q = tracker.update(c, s, p)
            if q is None:
                continue
            conf = tracker.smoother.confident.copy()
            ref = (tracker.smoother.reference.copy()
                   if tracker.smoother.reference is not None else None)
            r, okm, infos = ER.refine_quad(fr, q, conf, reference=ref)
            shift = float(np.linalg.norm(r - q, axis=1).max())
            rec = (t, fr.copy(), q.copy(), conf, ref)
            if conf.sum() == 3 and shift > 8:
                moves.append((shift, rec))
            if any(not inf.get("accepted") for inf in infos):
                rejects.append((shift, rec))
            if conf.sum() == 4 and shift < 6:
                cleans.append((shift, rec))
        cap.release()

        moves.sort(key=lambda x: -x[0])
        picks = [("move", r) for _, r in moves[:2]]
        if rejects:
            picks.append(("reject", rejects[len(rejects) // 2][1]))
        if cleans:
            picks.append(("clean", cleans[len(cleans) // 2][1]))
        print(f"=== {venue}: move {len(moves)} / reject {len(rejects)} / clean {len(cleans)} "
              f"→ 輸出 {len(picks)} 組 ===")
        for kind, (t, fr, q, conf, ref) in picks:
            tag = f"{venue}_{kind}_t{t:.2f}"
            print(f"  {tag}  被遮={[CN[i] for i in range(4) if not conf[i]]}")
            render_case(fr, q, conf, ref, tag, out)
    print(f"\n全部輸出於 {out}")


if __name__ == "__main__":
    main()
