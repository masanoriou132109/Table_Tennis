#!/usr/bin/env python3
"""評估 CV baseline: 對所有標註影格跑 detect_table_quad,與 GT 比角點像素誤差。

- 只在「可見」的 GT 角點上計誤差 (遮擋角的 GT 座標是估計值,不當精確答案)
- train / val 分開報告 (依 data/splits/)
- 疊圖輸出到 data/viz_baseline/: 綠 = 偵測,白 = GT

用法:
    python scripts/eval_baseline.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_cv import CORNER_NAMES, detect_table_quad  # noqa: E402

FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
VIZ_DIR = PROJECT_ROOT / "data" / "viz_baseline"


def load_split(name: str) -> set[str]:
    return set((SPLITS_DIR / f"{name}.txt").read_text().split())


def main() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": load_split("train"), "val": load_split("val")}
    results = {"train": {"errors": [], "detected": 0, "total": 0},
               "val": {"errors": [], "detected": 0, "total": 0}}
    failures = []

    for ann_path in sorted(ANNOTATIONS_DIR.glob("*.json")):
        split = "val" if ann_path.stem in splits["val"] else "train"
        ann = json.loads(ann_path.read_text())
        image = cv2.imread(str(FRAMES_DIR / ann["image"]))
        if image is None:
            continue
        results[split]["total"] += 1

        gt = np.array([ann["corners"][n] for n in CORNER_NAMES])
        visible = [ann["visibility"][n] for n in CORNER_NAMES]
        pred = detect_table_quad(image)

        viz = image.copy()
        cv2.polylines(viz, [gt.astype(np.int32)], True, (255, 255, 255), 2)  # GT 白
        if pred is not None:
            results[split]["detected"] += 1
            cv2.polylines(viz, [pred.astype(np.int32)], True, (0, 255, 0), 2)  # 偵測 綠
            per_corner = np.linalg.norm(pred - gt, axis=1)
            for i, name in enumerate(CORNER_NAMES):
                if visible[i]:
                    results[split]["errors"].append(per_corner[i])
                cv2.circle(viz, tuple(pred[i].astype(int)), 6, (0, 255, 0), -1)
        else:
            failures.append(ann_path.stem)
            cv2.putText(viz, "DETECTION FAILED", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(str(VIZ_DIR / (ann_path.stem + ".jpg")), viz,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    for split in ("train", "val"):
        r = results[split]
        errs = np.array(r["errors"])
        print(f"\n[{split}]  偵測率 {r['detected']}/{r['total']}")
        if len(errs):
            print(f"  可見角點誤差 (px): mean={errs.mean():.1f}  median={np.median(errs):.1f}  "
                  f"p95={np.percentile(errs, 95):.1f}  max={errs.max():.1f}")
            print(f"  <5px: {(errs < 5).mean():.0%}   <10px: {(errs < 10).mean():.0%}")

    if failures:
        print(f"\n偵測失敗 {len(failures)} 張:")
        for f in failures:
            print(f"  {f}")
    print(f"\n疊圖輸出於 {VIZ_DIR}")


if __name__ == "__main__":
    main()
