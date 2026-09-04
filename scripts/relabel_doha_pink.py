#!/usr/bin/env python3
"""用粉紅台面色塊重新預標 Doha 世錦賽影格 (取代 v1 的塌陷預標)。

Doha 桌面是鮮明粉紅色,v1 模型在此崩潰、CV magenta 邊線 baseline 也失效,
但直接抓粉紅台面色塊 + 擬合四邊形效果極佳 (37/37)。
覆寫 data/labelme_pool/ 中 Doha 影格的 polygon 角點,供 labelme 微調。

用法:
    python scripts/relabel_doha_pink.py [--prefix 20250523]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_cv import fit_quad, order_corners  # noqa: E402
from src.dataset import CORNER_NAMES  # noqa: E402

FRAMES_POOL = PROJECT_ROOT / "data" / "frames_pool"
LABELME_POOL = PROJECT_ROOT / "data" / "labelme_pool"

# 粉紅台面 HSV 範圍 (由 Doha 影格取樣而得)
PINK_LOWER = np.array([150, 60, 90])
PINK_UPPER = np.array([175, 255, 255])


def detect_pink_quad(image: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PINK_LOWER, PINK_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.01 * image.shape[0] * image.shape[1]:
        return None
    quad = fit_quad((labels == i).astype(np.uint8))
    return order_corners(quad) if quad is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="20250523", help="Doha 影格檔名前綴")
    args = parser.parse_args()

    jsons = sorted(LABELME_POOL.glob(f"{args.prefix}*.json"))
    updated = failed = 0
    for jp in jsons:
        image = cv2.imread(str(FRAMES_POOL / f"{jp.stem}.jpg"))
        quad = detect_pink_quad(image)
        if quad is None:
            failed += 1
            continue
        d = json.loads(jp.read_text())
        d["shapes"][0]["points"] = [[round(float(x), 1), round(float(y), 1)] for x, y in quad]
        jp.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        updated += 1

    print(f"角點順序: {CORNER_NAMES}")
    print(f"完成: 覆寫 {updated} 個,失敗 {failed} 個 (共 {len(jsons)})")


if __name__ == "__main__":
    main()
