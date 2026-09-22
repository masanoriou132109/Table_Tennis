#!/usr/bin/env python3
"""用訓練好的模型對影片做角點推論。

兩種輸出模式:
  --mode video   輸出疊了桌面四邊形的影片 (可指定 --start/--dur 只處理片段)
  --mode sample  跨整部影片均勻抽 N 幀,輸出疊圖到 data/infer_sample/<影片名>/

綠框 = presence>0.5 的預測;紅點 = 該角 heatmap 峰值低於 --score-thresh (低信心)。

用法:
    python scripts/infer_video.py "videos/xxx.mp4" --mode sample --n 8
    python scripts/infer_video.py "videos/xxx.mp4" --mode video --start 300 --dur 30
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

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, INPUT_H, INPUT_W  # noqa: E402
from src.homography import draw_table_grid  # noqa: E402
from src.edge_refine import refine_quad  # noqa: E402
from src.model import TableKeypointNet, decode_heatmaps  # noqa: E402
from src.tracker import TableTracker  # noqa: E402

CORNER_COLORS = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (255, 255, 0)]


def preprocess(frame: np.ndarray, device) -> torch.Tensor:
    img = cv2.resize(frame, (INPUT_W, INPUT_H))[..., ::-1].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)


@torch.no_grad()
def predict(model, frame, device, orig_w, orig_h):
    heatmaps, presence_logit, _ = model(preprocess(frame, device))
    coords, scores = decode_heatmaps(heatmaps)
    coords = coords[0].cpu().numpy() * [orig_w / INPUT_W, orig_h / INPUT_H]
    scores = scores[0].cpu().numpy()
    presence = float(torch.sigmoid(presence_logit)[0])
    return coords, scores, presence


def draw(frame, coords, scores, presence, score_thresh):
    if presence <= 0.5:
        cv2.putText(frame, f"NO TABLE ({presence:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
        return frame
    cv2.polylines(frame, [coords.astype(np.int32)], True, (0, 255, 0), 2)
    for i in range(4):
        color = CORNER_COLORS[i] if scores[i] >= score_thresh else (0, 0, 255)
        cv2.circle(frame, tuple(coords[i].astype(int)), 7, color, -1)
    cv2.putText(frame, f"table {presence:.2f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--ckpt", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--mode", choices=("video", "sample"), default="sample")
    parser.add_argument("--n", type=int, default=8, help="sample 模式抽幾幀")
    parser.add_argument("--start", type=float, default=0.0, help="video 模式起始秒")
    parser.add_argument("--dur", type=float, default=30.0, help="video 模式時長秒")
    parser.add_argument("--score-thresh", type=float, default=0.75)
    parser.add_argument("--smooth", action="store_true", help="video 模式啟用時序平滑")
    parser.add_argument("--grid", action="store_true", help="疊真實桌面座標網格 (homography)")
    parser.add_argument("--show-scores", action="store_true", help="每個角標註模型原始信心分數")
    parser.add_argument("--edge-refine", action="store_true",
                        help="用桌面邊線精修角點 (需 --smooth);遮擋角改由兩條邊線交點決定")
    parser.add_argument("--edge-alpha", type=float, default=0.5,
                        help="精修結果的時序平滑係數 (小=穩定但反應慢)")
    parser.add_argument("--out", default=None, help="輸出檔名 (video 模式)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = TableKeypointNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(args.video).stem

    if args.mode == "sample":
        out_dir = PROJECT_ROOT / "data" / "infer_sample" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        idxs = np.linspace(total * 0.1, total * 0.9, args.n).astype(int)
        for k, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            coords, scores, presence = predict(model, frame, device, w, h)
            draw(frame, coords, scores, presence, args.score_thresh)
            cv2.imwrite(str(out_dir / f"{k:02d}_f{fi}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"抽樣疊圖輸出於 {out_dir}")
    else:
        out_path = Path(args.out) if args.out else PROJECT_ROOT / "data" / f"infer_{stem}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        start_f = int(args.start * fps)
        end_f = min(total, int((args.start + args.dur) * fps))
        tracker = TableTracker(w, h, score_thresh=args.score_thresh) if args.smooth else None
        edge_state = None   # 精修結果的 EMA 狀態 (抑制逐幀抖動)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(end_f - start_f):
            ok, frame = cap.read()
            if not ok:
                break
            coords, scores, presence = predict(model, frame, device, w, h)
            raw_scores = scores.copy()  # 模型原始分數 (供標註)
            if tracker is not None:
                quad = tracker.update(coords, scores, presence)
                if quad is not None:  # 通過三重閘門且已確認
                    coords = quad
                    scores = np.where(tracker.smoother.confident, scores, 0.0)  # 補算角畫紅點
                    presence = 1.0
                else:
                    presence = 0.0  # 未確認/被閘門擋下 → 不繪製
                    edge_state = None
            if args.edge_refine and tracker is not None and presence > 0.5:
                refined, _, _ = refine_quad(frame, coords, tracker.smoother.confident,
                                            reference=tracker.smoother.reference)
                edge_state = refined if edge_state is None else \
                    (1 - args.edge_alpha) * edge_state + args.edge_alpha * refined
                coords = edge_state
            if args.grid and presence > 0.5:
                draw_table_grid(frame, coords)
            draw(frame, coords, scores, presence, args.score_thresh)
            if args.show_scores and presence > 0.5:
                for i in range(4):
                    col = CORNER_COLORS[i] if scores[i] >= args.score_thresh else (0, 0, 255)
                    x, y = int(coords[i][0]), int(coords[i][1])
                    cv2.putText(frame, f"{raw_scores[i]:.2f}", (x + 8, y + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            writer.write(frame)
        writer.release()
        print(f"疊框影片輸出於 {out_path}  (smooth={'on' if tracker else 'off'})")

    cap.release()


if __name__ == "__main__":
    main()
