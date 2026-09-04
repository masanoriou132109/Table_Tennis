#!/usr/bin/env python3
"""把 labelme 標註轉成本專案的角點 JSON 格式 (data/annotations/)。

labelme 標註約定:
  - 每張圖一個 label 為 "table" 的 polygon,恰好 4 個點 (點擊順序不限,會自動重排)
  - flags 勾選 far_left / far_right / near_right / near_left = 該角被遮擋或出畫面
    (轉換後 visibility 為 false)

建議的 labelme 啟動指令:
  labelme data/frames --output data/labelme \
    --flags far_left,far_right,near_right,near_left --nodata --autosave

用法:
    python scripts/convert_labelme.py [--input data/labelme]
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "labelme"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]


def order_corners(pts: np.ndarray) -> np.ndarray:
    """把 4 點排成固定語意順序: 遠左, 遠右, 近右, 近左 (遠 = 畫面上方)。
    與 annotate_sam.py 的排序邏輯一致。"""
    pts = pts[np.argsort(pts[:, 1])]  # 依 y 排序
    far, near = pts[:2], pts[2:]
    far = far[np.argsort(far[:, 0])]
    near = near[np.argsort(near[:, 0])]
    return np.array([far[0], far[1], near[1], near[0]])


def convert_one(labelme_path: Path) -> bool:
    data = json.loads(labelme_path.read_text())

    table_shapes = [s for s in data.get("shapes", [])
                    if s.get("label") == "table" and s.get("shape_type") == "polygon"]
    if not table_shapes:
        print(f"  [跳過] {labelme_path.name}: 沒有 label 為 'table' 的 polygon")
        return False
    points = table_shapes[0]["points"]
    if len(points) != 4:
        print(f"  [跳過] {labelme_path.name}: polygon 有 {len(points)} 點,需要恰好 4 點")
        return False

    corners = order_corners(np.array(points, dtype=np.float64))
    flags = data.get("flags", {})  # 勾選 = 遮擋 -> visibility False

    out = {
        "image": Path(data.get("imagePath", labelme_path.stem + ".jpg")).name,
        "image_width": data["imageWidth"],
        "image_height": data["imageHeight"],
        "corner_order": CORNER_NAMES,
        "corners": {name: [round(float(x), 1), round(float(y), 1)]
                    for name, (x, y) in zip(CORNER_NAMES, corners)},
        "visibility": {name: not flags.get(name, False) for name in CORNER_NAMES},
    }
    out_path = ANNOTATIONS_DIR / (labelme_path.stem + ".json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"labelme JSON 目錄 (預設 {DEFAULT_INPUT})")
    args = parser.parse_args()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    labelme_files = sorted(args.input.glob("*.json"))
    if not labelme_files:
        print(f"在 {args.input} 找不到 labelme JSON")
        return

    converted = sum(convert_one(p) for p in labelme_files)
    print(f"\n完成: {converted}/{len(labelme_files)} 個標註轉換到 {ANNOTATIONS_DIR}")


if __name__ == "__main__":
    main()
