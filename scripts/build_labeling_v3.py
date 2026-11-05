#!/usr/bin/env python3
"""v3 標註池: 挑「角被球員擋住 / 模型沒把握」的主鏡頭影格, 用離線平滑的框預標, 給 labelme 修正。

v2 (216 張) 只有 4% 的角是被擋的, 而且被擋的角在訓練時沒有 heatmap 目標 → 模型學到「被擋就放棄」,
回合中 66% 的主鏡頭影格有角信心 < 0.9, 球員衣服還會被誤認成桌角。v3 補這類影格。

流程 (每部影片):
  1. 全片每 --stride 幀跑一次桌面模型 (Core ML), 結果快取在 data/cache/table_scan_*.npz
  2. TableTracker + 離線前後平滑 (src/offline_smooth.py) → 每幀的預標框
  3. 候選 = 主鏡頭有框的幀, 排除 regression 測試片段 (前後 5 秒, 留著當測試集)
  4. 時間軸切成 quota 段, 每段挑一幀: 4/5 的段挑「最難」(信心 < 0.9 的角最多, 再比最低信心),
     1/5 挑四角都 >= 0.9 的一般幀 (別只學難的); 相鄰挑選 >= 2 秒
  5. 影格縮成 1280x720 存 data/frames_v3/, labelme JSON 存 data/labelme_v3/:
     polygon = 平滑後的框; flags 預先勾「模型信心 < 0.5」的角 (疑似被擋, 請確認)

標註約定 (與 v2 不同處): 被擋的角也要點在「估計的真實位置」, 並勾該角的 flag。

用法:
    python scripts/build_labeling_v3.py "videos/highqua_videos/xxx.mp4:60" "...:50" ...
    (冒號後是該影片要挑的張數)
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

from scripts.extract_frames import short_stem  # noqa: E402
from scripts.infer_video import load_table_model, predict  # noqa: E402
from src.dataset import CORNER_NAMES  # noqa: E402
from src.offline_smooth import smooth_quads  # noqa: E402
from src.tracker import TableTracker  # noqa: E402
from src.video_io import PROC_H, PROC_W, to_proc  # noqa: E402

FRAMES_DIR = PROJECT_ROOT / "data" / "frames_v3"
LABELME_DIR = PROJECT_ROOT / "data" / "labelme_v3"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
TRUST = 0.9          # 與離線平滑的可信門檻相同
FLAG_BELOW = 0.5     # 預標時先勾「疑似被擋」的信心門檻


def scan(video: Path, stride: int, model, device) -> dict:
    """全片每 stride 幀跑桌面模型 (快取)。回傳 frame / coords / scores / presence / fps。"""
    cache = CACHE_DIR / f"table_scan_{short_stem(video)}_st{stride}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        if fi % 3000 < stride:
            print(f"  {video.name[:30]}: {fi}/{total}", flush=True)
    cap.release()
    out = {"frame": np.array(F), "coords": np.array(C, np.float64), "scores": np.array(S, np.float64),
           "presence": np.array(P, np.float64), "fps": np.array(fps)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def regression_windows(video: Path, pad: float = 5.0) -> list[tuple[float, float]]:
    spec = json.loads((PROJECT_ROOT / "tests" / "regression_cases.json").read_text())
    return [(c["start"] - pad, c["start"] + c["dur"] + pad) for c in spec["clips"].values()
            if Path(c["video"]).name == video.name]


def to_labelme(frame_name: str, quad: np.ndarray, flags: list[bool]) -> dict:
    return {
        "version": "5.0.0",
        "flags": {n: bool(f) for n, f in zip(CORNER_NAMES, flags)},   # 勾選 = 該角被擋
        "shapes": [{"label": "table",
                    "points": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
                    "group_id": None, "shape_type": "polygon", "flags": {}}],
        "imagePath": f"../frames_v3/{frame_name}",
        "imageData": None, "imageHeight": PROC_H, "imageWidth": PROC_W,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="影片路徑:張數")
    ap.add_argument("--stride", type=int, default=2, help="每幾幀跑一次模型 (平滑用的時間解析度)")
    ap.add_argument("--min-gap", type=float, default=2.0, help="挑出的幀至少相隔幾秒")
    ap.add_argument("--easy-every", type=int, default=5, help="每幾段挑一張一般幀")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_table_model(str(PROJECT_ROOT / "checkpoints" / "best.pt"), device, "coreml")
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    LABELME_DIR.mkdir(parents=True, exist_ok=True)

    for spec in args.videos:
        path, quota = spec.rsplit(":", 1)
        video, quota = Path(path), int(quota)
        z = scan(video, args.stride, model, device)
        fps = float(z["fps"]) / args.stride                     # 取樣後的有效幀率
        fr = z["frame"].tolist()
        seq = list(range(len(fr)))                              # 平滑在「取樣序號」上做
        tr = TableTracker(PROC_W, PROC_H)
        tq = {}
        for k in seq:
            q = tr.update(z["coords"][k], z["scores"][k], float(z["presence"][k]))
            if q is not None:
                tq[k] = q
        sq = smooth_quads(seq, z["coords"], z["scores"], tq, fps)
        excl = regression_windows(video)
        real_fps = float(z["fps"])
        cand = [k for k in sq if not any(a <= fr[k] / real_fps <= b for a, b in excl)]
        if not cand:
            print(f"{video.name}: 沒有候選幀"); continue
        sc = z["scores"]
        hard = {k: (int((sc[k] < TRUST).sum()), -float(sc[k].min())) for k in cand}
        cand.sort()
        t_cand = np.array([fr[k] / real_fps for k in cand])
        edges = np.linspace(t_cand[0], t_cand[-1] + 1e-6, quota + 1)
        picked: list[int] = []
        for b in range(quota):
            pool = [cand[i] for i in np.where((t_cand >= edges[b]) & (t_cand < edges[b + 1]))[0]
                    if all(abs(fr[cand[i]] - fr[p]) / real_fps >= args.min_gap for p in picked)]
            if not pool:
                continue
            if (b + 1) % args.easy_every == 0:
                easy = [k for k in pool if hard[k][0] == 0]
                picked.append(int(rng.choice(easy)) if easy else max(pool, key=lambda k: hard[k]))
            else:
                picked.append(max(pool, key=lambda k: hard[k]))

        cap = cv2.VideoCapture(str(video))
        n_hard = 0
        for k in sorted(picked):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr[k])
            ok, frame = cap.read()
            if not ok:
                continue
            name = f"v3_{short_stem(video, 24)}_f{fr[k]:06d}.jpg"
            cv2.imwrite(str(FRAMES_DIR / name), to_proc(frame), [cv2.IMWRITE_JPEG_QUALITY, 95])
            flags = (sc[k] < FLAG_BELOW).tolist()
            (LABELME_DIR / name.replace(".jpg", ".json")).write_text(
                json.dumps(to_labelme(name, sq[k], flags), ensure_ascii=False, indent=1))
            n_hard += hard[k][0] > 0
        cap.release()
        print(f"{video.name}: 挑 {len(picked)} 張 (有角信心 < 0.9 的 {n_hard} 張), 候選 {len(cand)} 幀")


if __name__ == "__main__":
    main()
