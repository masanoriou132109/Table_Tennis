#!/usr/bin/env python3
"""完整示範影片: 桌面偵測 + edge 精修 + 強證據覆寫可信角。

主畫面就是實際會出貨的樣子 (橘框 = 最終結果, 含 EMA 時序平滑)。
另外因為覆寫只在約 1% 的幀觸發、一幀只有 1/50 秒, 直接看會完全看不到,
所以觸發時會:
  - 把「舊行為」(override 關閉) 的框用綠色疊上去, 並保持 --hold 秒
  - 左上角顯示 OVERRIDE <角> <位移>, 同樣保持 --hold 秒
這樣可以逐次確認每個覆寫是救對了還是弄歪了。

用法:
    python scripts/demo_override.py VIDEO --start 88 --dur 90 --out data/xxx.mp4
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=60.0)
    ap.add_argument("--edge-alpha", type=float, default=0.3, help="精修結果的 EMA 係數")
    ap.add_argument("--hold", type=float, default=1.5, help="覆寫發生後對照框保持幾秒")
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=None)
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    out_path = Path(args.out) if args.out else \
        PROJECT_ROOT / "data" / f"demo_override_{Path(args.video).stem[:24]}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = TableTracker(w, h)
    ema_new = ema_old = None
    hold_left, hold_msg, n_fire = 0, "", 0

    for i in range(int(args.dur * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + i / fps
        c, s, p = predict(model, frame, device, w, h)
        quad = tracker.update(c, s, p)

        if quad is None:
            ema_new = ema_old = None
            cv2.putText(frame, f"t={t:.2f}s   no table", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 230), 2)
            writer.write(frame)
            continue

        conf = tracker.smoother.confident.copy()
        ref = tracker.smoother.reference
        q_new, _, _ = ER.refine_quad(frame, quad, conf, reference=ref, strong_override=True)
        q_old, _, _ = ER.refine_quad(frame, quad, conf, reference=ref, strong_override=False)
        a = args.edge_alpha
        ema_new = q_new if ema_new is None else (1 - a) * ema_new + a * q_new
        ema_old = q_old if ema_old is None else (1 - a) * ema_old + a * q_old

        d = np.linalg.norm(q_new - q_old, axis=1)
        if d.max() > 0.5:
            j = int(d.argmax())
            n_fire += 1
            hold_left = int(args.hold * fps)
            hold_msg = f"OVERRIDE  {CN[j]}  {d[j]:.1f}px   (score {s[j]:.2f})"

        if hold_left > 0:   # 對照: 舊行為的框
            cv2.polylines(frame, [ema_old.astype(np.int32)], True, (0, 255, 0), 2)
            hold_left -= 1

        cv2.polylines(frame, [ema_new.astype(np.int32)], True, (0, 180, 255), 2)
        for k in range(4):
            cv2.circle(frame, tuple(ema_new[k].astype(int)), 6,
                       (0, 180, 255) if conf[k] else (0, 0, 255), -1)

        cv2.putText(frame, f"t={t:.2f}s   orange = shipped (edge refine + override)",
                    (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if hold_msg and hold_left > 0:
            cv2.putText(frame, hold_msg, (25, 78), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 180, 255), 2)
            cv2.putText(frame, "green = old behaviour (no override)", (25, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"輸出 {out_path}   覆寫觸發 {n_fire} 次")


if __name__ == "__main__":
    main()
