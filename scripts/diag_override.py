#!/usr/bin/env python3
"""診斷「強證據覆寫可信角」: 救到的是真的壞角, 還是把本來就對的角弄歪了?

同一個追蹤器輸出的四邊形, 只把 refine_quad 的 strong_override 切換:
  綠框 = strong_override=False (舊行為: 可信角最多只動 4px)
  橘框 = strong_override=True  (新行為: 邊線證據強時放寬到 shift_strong)
因為兩次呼叫吃同一個 quad 與同一個 seed, 差異純粹來自覆寫, 沒有狀態分歧。

輸出兩類影格供人工檢查:
  fire  = 覆寫有生效的幀, 按位移由大到小 → 看橘框是否真的比綠框貼桌
  quiet = 未覆寫但四角全可信的幀        → 看橘框有沒有被無故改動 (應與綠框重合)

同時印出觸發率與位移分佈: 若觸發率很高或位移普遍只有 5~6px,
代表門檻太鬆, 是在對本來就準的角做無意義的搬動。

用法:
    python scripts/diag_override.py [--dur 60] [--per-video 3]
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

CN = ["FL", "FR", "NR", "NL"]
EDGE_NAMES = ["far", "right", "near", "left"]

VIDEOS = [
    ("Incheon", "videos/2025WTTInche-Lee Sang Su.webm", 88),
    ("Doha", "videos/20250523World Table Tennis Championships-Wang Chuqin.mp4", 200),
    ("Vegas", "videos/20250712USA-Tomokazu Harimoto.mp4", 100),
    ("Chongqing", "videos/20250312Chongqing-Jang Woojin.mp4", 116),
    ("IncheonSora", "videos/Incheon Sora Matsushima 2025.mp4", 120),
]


def render(frame, quad, q_old, q_new, conf, scores, infos, tag, out: Path):
    vis = frame.copy()
    cv2.polylines(vis, [q_old.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(vis, [q_new.astype(np.int32)], True, (0, 180, 255), 2)
    for i in range(4):
        cv2.circle(vis, tuple(q_old[i].astype(int)), 6, (0, 255, 0), -1)
        cv2.circle(vis, tuple(q_new[i].astype(int)), 9, (0, 180, 255), 2)
    # 被覆寫的角: 用 edge_strong 的那幾條邊的內點畫出來, 讓人直接看證據
    for i, info in enumerate(infos):
        if not info.get("strong"):
            continue
        inl = info["inlier_mask"]
        for k, p in enumerate(info["kept_pts"]):
            if inl is not None and inl[k]:
                cv2.circle(vis, tuple(np.round(p).astype(int)), 2, (0, 255, 255), -1)

    lines = [f"{tag}   green = override OFF (old)   orange = override ON (new)",
             "score:  " + "  ".join(
                 f"{CN[i]}{scores[i]:.2f}{'' if conf[i] else '(occ)'}" for i in range(4)),
             "edges:  " + "  ".join(
                 f"{EDGE_NAMES[i]}[{'S' if infos[i].get('strong') else ('A' if infos[i].get('accepted') else '-')}"
                 f" n{infos[i]['n_inliers']} sp{infos[i]['span']:.0f}/{infos[i]['length']:.0f}"
                 f" {infos[i].get('angle_dev', float('nan')):.1f}d]" for i in range(4))]
    moved = [f"{CN[i]} {np.linalg.norm(q_new[i] - q_old[i]):.1f}px"
             for i in range(4) if np.linalg.norm(q_new[i] - q_old[i]) > 0.5]
    lines.append("OVERRIDDEN: " + (", ".join(moved) if moved else "none")
                 + "    yellow dots = strong-edge inliers")
    for k, s in enumerate(lines):
        cv2.putText(vis, s, (18, 30 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 180, 255) if k == len(lines) - 1 and moved else (255, 255, 255),
                    2, cv2.LINE_AA)
    cv2.imwrite(str(out / f"{tag}_override.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dur", type=float, default=60.0, help="每部影片掃描秒數")
    ap.add_argument("--per-video", type=int, default=3, help="每部影片輸出幾組觸發幀")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "diag_override"))
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

        fires, quiets, n_frames = [], [], 0
        for i in range(int(args.dur * fps)):
            ok, fr = cap.read()
            if not ok:
                break
            t = start_s + i / fps
            c, s, p = predict(model, fr, device, w, h)
            quad = tracker.update(c, s, p)
            if quad is None:
                continue
            n_frames += 1
            conf = tracker.smoother.confident.copy()
            ref = tracker.smoother.reference
            q_old, _, _ = ER.refine_quad(fr, quad, conf, reference=ref,
                                         strong_override=False)
            q_new, _, infos = ER.refine_quad(fr, quad, conf, reference=ref,
                                             strong_override=True)
            d = np.linalg.norm(q_new - q_old, axis=1)
            rec = (t, fr.copy(), quad.copy(), q_old, q_new, conf, s.copy(), infos)
            if d.max() > 0.5:
                fires.append((float(d.max()), rec))
            elif conf.all():
                quiets.append(rec)
        cap.release()

        rate = len(fires) / max(n_frames, 1) * 100
        mags = np.array([m for m, _ in fires]) if fires else np.zeros(0)
        print(f"=== {venue}: 有框 {n_frames} 幀, 覆寫生效 {len(fires)} 幀 ({rate:.1f}%) ===")
        if len(mags):
            print("    位移分佈 (px): " + "  ".join(
                f"p{q}={np.percentile(mags, q):.1f}" for q in [50, 75, 90, 100]))
        fires.sort(key=lambda x: -x[0])
        for _, rec in fires[:args.per_video]:
            t, fr, quad, qo, qn, conf, sc, infos = rec
            j = int(np.linalg.norm(qn - qo, axis=1).argmax())
            tag = f"{venue}_fire_t{t:.2f}_{CN[j]}"
            print(f"  {tag}  位移{np.linalg.norm(qn[j] - qo[j]):.1f}px  分數{sc[j]:.2f}  "
                  f"強邊={[EDGE_NAMES[k] for k in range(4) if infos[k].get('strong')]}")
            render(fr, quad, qo, qn, conf, sc, infos, tag, out)
        if quiets:  # 未觸發的對照: 橘綠應完全重合
            rec = quiets[len(quiets) // 2]
            t, fr, quad, qo, qn, conf, sc, infos = rec
            tag = f"{venue}_quiet_t{t:.2f}"
            print(f"  {tag}  未覆寫, 橘綠最大差 {np.linalg.norm(qn - qo, axis=1).max():.2f}px")
            render(fr, quad, qo, qn, conf, sc, infos, tag, out)

    print(f"\n全部輸出於 {out}")


if __name__ == "__main__":
    main()
