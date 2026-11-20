#!/usr/bin/env python3
"""v4 負樣本池: 挑「v3 說有桌面, 但其實是側拍 / 重播」的影格, 給使用者確認後當 presence = 0 訓練。

v3 在側拍/重播畫面 presence 偏高 (London 10:03 段 >= 0.5 從 v2 的 18% 變 100%), 「側拍不借桌面框」
失效, 球模型在側拍抓到的假球被映射成擊球/落點。原訓練集只有 27 張負樣本 (都是舊短片的球員特寫)。

流程 (每部影片):
  1. 全片每 --step 秒跑一次目前的桌面模型 (v3, Core ML), 快取 data/cache/table_scan_v3_*.npz
  2. 主鏡頭參考框 = 四角信心都 >= 0.9 且 presence >= 0.9 的框的中位數 (主鏡頭整場位置只差 5-20px)
  3. 疑似非主鏡頭 = presence >= 0.5 且 (框不合理 或 與參考框的平均角距離 > --dist px)
  4. 只取連續 >= --min-run 秒都疑似的段 (排除轉場閃一下), 排除 regression 片段前後 5 秒
  5. 每段最多挑 --per-run 張 (段內平均分布), 依時間分 quota 份平均取
  6. 存 1280x720 到 data/frames_v4neg/ + 縮圖總表 data/frames_v4neg/_sheet_*.jpg 給使用者挑掉主鏡頭

使用者確認後 (刪掉其實是主鏡頭的), 搬進 data/frames/ (不附標註 JSON = 負樣本)。

用法:
    python scripts/build_negatives_v4.py "videos/highqua_videos/xxx.mp4:50" ...
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

from scripts.build_labeling_v3 import regression_windows  # noqa: E402
from scripts.extract_frames import short_stem  # noqa: E402
from scripts.infer_video import load_table_model, predict  # noqa: E402
from src.geometry import is_plausible_quad  # noqa: E402
from src.video_io import PROC_H, PROC_W, to_proc  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "frames_v4neg"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"


def scan(video: Path, step_s: float, model, device) -> dict:
    """全片每 step_s 秒跑桌面模型 (快取)。"""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(step_s * fps))
    cache = CACHE_DIR / f"table_scan_v3_{short_stem(video)}_st{stride}.npz"
    if cache.exists():
        cap.release()
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    F, C, S, P = [], [], [], []
    fi = 0
    while True:
        if fi % stride:
            if not cap.grab():
                break
            fi += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        coords, scores, presence = predict(model, to_proc(frame), device, PROC_W, PROC_H)
        F.append(fi); C.append(coords); S.append(scores); P.append(presence)
        fi += 1
        if fi % 6000 < stride:
            print(f"  {video.name[:30]}: {fi}/{total}", flush=True)
    cap.release()
    out = {"frame": np.array(F), "coords": np.array(C, np.float64), "scores": np.array(S, np.float64),
           "presence": np.array(P, np.float64), "fps": np.array(fps)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def suspect_mask(z: dict, dist_px: float) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (疑似非主鏡頭, 與主鏡頭參考框的平均角距離)。"""
    C, S, P = z["coords"], z["scores"], z["presence"]
    main = (S.min(1) >= 0.9) & (P >= 0.9)
    ref = np.median(C[main], axis=0)
    dist = np.linalg.norm(C - ref, axis=2).mean(1)
    bad = np.array([not is_plausible_quad(q, PROC_W, PROC_H) for q in C])
    return (P >= 0.5) & (bad | (dist > dist_px)), dist


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """連續 True 的區段 [a, b) (取樣序號)。"""
    out, a = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i)); a = None
    return out


def contact_sheet(paths: list[Path], out: Path, cols: int = 6, w: int = 320) -> None:
    h = w * 9 // 16
    for page in range(0, len(paths), cols * 6):
        chunk = paths[page:page + cols * 6]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.zeros((rows * (h + 22), cols * w, 3), np.uint8)
        for i, p in enumerate(chunk):
            r, c = divmod(i, cols)
            im = cv2.resize(cv2.imread(str(p)), (w, h))
            y = r * (h + 22)
            sheet[y:y + h, c * w:(c + 1) * w] = im
            cv2.putText(sheet, f"{page + i + 1:03d} {p.stem[-14:]}", (c * w + 4, y + h + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out.with_name(f"{out.stem}_{page // (cols * 6) + 1}.jpg")), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="影片路徑:張數")
    ap.add_argument("--step", type=float, default=0.2, help="掃描間隔 (秒)")
    ap.add_argument("--dist", type=float, default=80.0, help="與主鏡頭參考框的平均角距離門檻 (px, 1280x720)")
    ap.add_argument("--min-run", type=float, default=1.0, help="疑似段至少持續幾秒")
    ap.add_argument("--per-run", type=int, default=2, help="每段最多挑幾張")
    ap.add_argument("--scan-only", action="store_true", help="只掃描並印統計")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_table_model(str(PROJECT_ROOT / "checkpoints" / "best.pt"), device, "coreml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = []
    for spec in args.videos:
        path, quota = spec.rsplit(":", 1)
        video, quota = Path(path), int(quota)
        z = scan(video, args.step, model, device)
        fps = float(z["fps"])
        fr = z["frame"]
        t = fr / fps
        sus, dist = suspect_mask(z, args.dist)
        excl = regression_windows(video)
        in_excl = np.array([any(a <= x <= b for a, b in excl) for x in t])
        rs = [(a, b) for a, b in runs(sus & ~in_excl) if t[b - 1] - t[a] >= args.min_run]
        print(f"{video.name}: 掃 {len(fr)} 幀, presence>=0.5 {np.mean(z['presence'] >= 0.5):.0%}, "
              f"疑似非主鏡頭 {sus.mean():.1%}, 合格段 {len(rs)} (共 {sum(t[b - 1] - t[a] for a, b in rs):.0f} 秒)")
        if args.scan_only or not rs:
            continue
        # 每段平均取 per-run 張 (去掉段首尾各 0.3 秒, 避開轉場)
        cand = []
        for a, b in rs:
            ks = [k for k in range(a, b) if t[a] + 0.3 <= t[k] <= t[b - 1] - 0.3] or list(range(a, b))
            n = min(args.per_run, len(ks))
            cand.extend(ks[int(i)] for i in np.linspace(0, len(ks) - 1, n + 2)[1:-1].round())
        cand = sorted(set(cand))
        if len(cand) > quota:                                 # 依時間平均抽 quota 張 (去重)
            cand = sorted({cand[int(i)] for i in np.linspace(0, len(cand) - 1, quota).round()})
        cap = cv2.VideoCapture(str(video))
        for k in cand:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[k]))
            ok, frame = cap.read()
            if not ok:
                continue
            p = OUT_DIR / f"v4neg_{short_stem(video, 24)}_f{int(fr[k]):06d}.jpg"
            cv2.imwrite(str(p), to_proc(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
            all_paths.append(p)
        cap.release()
        print(f"  挑 {len(cand)} 張")

    if all_paths:
        contact_sheet(all_paths, OUT_DIR / "_sheet.jpg")
        print(f"共 {len(all_paths)} 張, 縮圖總表 {OUT_DIR}/_sheet_*.jpg")


if __name__ == "__main__":
    main()
