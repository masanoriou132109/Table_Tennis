#!/usr/bin/env python3
"""從 videos/ 中的轉播影片抽格,存到 data/frames/。

- 依固定時間間隔取樣 (預設每 0.5 秒一格)
- 以縮圖灰階差異過濾幾乎重複的影格,確保資料多樣性
- 檔名格式: <影片檔名前綴>_f<影格序號>.jpg

用法:
    python scripts/extract_frames.py [--interval 0.5] [--diff-thresh 8.0]
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "videos"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def short_stem(video_path: Path, max_len: int = 40) -> str:
    """影片檔名很長,取前綴並移除不適合當檔名的字元。"""
    stem = re.sub(r"[^\w一-鿿]+", "_", video_path.stem)
    return stem[:max_len].rstrip("_")


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """縮小成 32x32 灰階,當作比較影格差異的簽名。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32)).astype(np.float32)


def extract_video(video_path: Path, interval_sec: float, diff_thresh: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [跳過] 無法開啟: {video_path.name}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * interval_sec))
    stem = short_stem(video_path)

    saved = 0
    prev_sig = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            sig = frame_signature(frame)
            # 與上一張已保存影格的平均像素差,太小視為重複
            if prev_sig is None or float(np.abs(sig - prev_sig).mean()) >= diff_thresh:
                out_path = FRAMES_DIR / f"{stem}_f{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                prev_sig = sig
                saved += 1
        frame_idx += 1

    cap.release()
    print(f"  {video_path.name[:50]}...  -> {saved} 格")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.5, help="取樣間隔秒數 (預設 0.5)")
    parser.add_argument("--diff-thresh", type=float, default=8.0,
                        help="與上一張保存影格的平均灰階差門檻,低於此值視為重複 (預設 8.0)")
    args = parser.parse_args()

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in VIDEOS_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"在 {VIDEOS_DIR} 找不到影片")
        return

    total = 0
    for video in videos:
        total += extract_video(video, args.interval, args.diff_thresh)
    print(f"\n完成: {len(videos)} 部影片,共 {total} 格,輸出於 {FRAMES_DIR}")


if __name__ == "__main__":
    main()
