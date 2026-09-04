#!/usr/bin/env python3
"""把 data/annotations/ 的角點標註畫回影格,輸出到 data/viz/ 供肉眼檢查。

實心圓 = 可見角點,空心圓 = 標為遮擋/出畫面的角點。
編號: 1=遠左 2=遠右 3=近右 4=近左

用法:
    python scripts/visualize_annotations.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
VIZ_DIR = PROJECT_ROOT / "data" / "viz"

CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]
CORNER_COLORS = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (255, 255, 0)]  # BGR


def main() -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    ann_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    if not ann_files:
        print(f"在 {ANNOTATIONS_DIR} 找不到標註")
        return

    occluded_count = 0
    for ann_path in ann_files:
        ann = json.loads(ann_path.read_text())
        img_path = FRAMES_DIR / ann["image"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [跳過] 找不到影像: {ann['image']}")
            continue

        pts = np.array([ann["corners"][n] for n in CORNER_NAMES], np.int32)
        cv2.polylines(image, [pts], True, (255, 255, 255), 2)
        has_occluded = False
        for i, name in enumerate(CORNER_NAMES):
            visible = ann["visibility"][name]
            has_occluded |= not visible
            filled = -1 if visible else 3  # 空心 = 遮擋
            cv2.circle(image, tuple(pts[i]), 10, CORNER_COLORS[i], filled)
            cv2.putText(image, str(i + 1), (pts[i][0] + 12, pts[i][1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, CORNER_COLORS[i], 2)
        if has_occluded:
            occluded_count += 1

        cv2.imwrite(str(VIZ_DIR / ann_path.with_suffix(".jpg").name), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"完成: {len(ann_files)} 張視覺化輸出於 {VIZ_DIR}")
    print(f"其中 {occluded_count} 張含遮擋角點")


if __name__ == "__main__":
    main()
