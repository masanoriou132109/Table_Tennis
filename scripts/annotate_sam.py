#!/usr/bin/env python3
"""SAM2 輔助的桌面四角標註工具。

流程:
  1. 逐張載入 data/frames/ 中尚未標註的影格
  2. 左鍵點桌面 (可多點) -> SAM2 分割 -> 自動擬合四邊形當初始角點
  3. 拖曳角點微調;角點可拖到畫面外 (視窗有留邊)
  4. 按 1-4 切換各角可見性 (角被遮擋或出畫面時標為不可見)
  5. 按 s 儲存 JSON 到 data/annotations/

角點語意順序固定為: far_left(遠左), far_right(遠右), near_right(近右), near_left(近左)
「遠」= 畫面上方那側。此順序是之後計算 homography 的依據,不可打亂。

操作鍵:
  左鍵點擊     加入 SAM 正向提示點 (點在桌面上)
  右鍵點擊     加入 SAM 負向提示點 (點在誤含的區域上)
  拖曳角點     微調角點位置
  1/2/3/4     切換 遠左/遠右/近右/近左 的可見性
  s           儲存並跳下一張
  n           跳過這張 (不儲存)
  r           重設本張
  q / ESC     離開

用法:
    python scripts/annotate_sam.py [--model sam2.1_b.pt] [--margin 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]
CORNER_LABELS_ZH = ["遠左", "遠右", "近右", "近左"]
CORNER_COLORS = [(0, 255, 255), (0, 165, 255), (255, 0, 255), (255, 255, 0)]  # BGR
DRAG_RADIUS = 15  # 抓取角點的像素半徑 (顯示座標系)


def fit_quad(mask: np.ndarray) -> np.ndarray | None:
    """從二值 mask 擬合凸四邊形,回傳 4x2 座標;失敗回傳 None。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    # 逐步放大 epsilon 直到剛好簡化成 4 點
    for eps_ratio in np.arange(0.01, 0.15, 0.01):
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
    # 簡化不出 4 點時退而用最小面積外接矩形
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return box.astype(np.float64)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """把 4 點排成固定語意順序: 遠左, 遠右, 近右, 近左 (遠 = 畫面上方)。"""
    pts = pts[np.argsort(pts[:, 1])]  # 依 y 排序
    far, near = pts[:2], pts[2:]
    far = far[np.argsort(far[:, 0])]    # 遠側依 x: 左, 右
    near = near[np.argsort(near[:, 0])]
    return np.array([far[0], far[1], near[1], near[0]])


class Annotator:
    def __init__(self, model_name: str, margin: int):
        from ultralytics import SAM  # 延後 import,啟動較快
        print(f"載入 SAM 模型 {model_name} ...")
        self.sam = SAM(model_name)
        self.margin = margin  # 畫布留邊,讓角點可以拖出影像範圍
        self.reset()

    def reset(self):
        self.corners: np.ndarray | None = None   # 4x2, 影像座標 (可為負或超出邊界)
        self.visibility = [True] * 4
        self.pos_points: list[list[int]] = []    # SAM 正向提示點
        self.neg_points: list[list[int]] = []    # SAM 負向提示點
        self.mask: np.ndarray | None = None
        self.drag_idx: int | None = None

    def run_sam(self, image: np.ndarray):
        points = self.pos_points + self.neg_points
        labels = [1] * len(self.pos_points) + [0] * len(self.neg_points)
        results = self.sam(image, points=[points], labels=[labels], verbose=False)
        masks = results[0].masks
        if masks is None or len(masks.data) == 0:
            print("SAM 沒有回傳 mask,請再點一次")
            return
        self.mask = masks.data[0].cpu().numpy() > 0.5
        quad = fit_quad(self.mask)
        if quad is not None:
            self.corners = order_corners(quad)

    # ---- 座標轉換: 顯示畫布 = 影像四周加 margin 留邊 ----
    def to_canvas(self, pt):
        return int(pt[0] + self.margin), int(pt[1] + self.margin)

    def to_image(self, x, y):
        return float(x - self.margin), float(y - self.margin)

    def on_mouse(self, event, x, y, flags, _param):
        ix, iy = self.to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 先檢查是否抓到既有角點 (拖曳優先於加提示點)
            if self.corners is not None:
                for i, c in enumerate(self.corners):
                    cx, cy = self.to_canvas(c)
                    if (cx - x) ** 2 + (cy - y) ** 2 <= DRAG_RADIUS ** 2:
                        self.drag_idx = i
                        return
            self.pos_points.append([int(ix), int(iy)])
            self.pending_sam = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.neg_points.append([int(ix), int(iy)])
            self.pending_sam = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx is not None:
            self.corners[self.drag_idx] = [ix, iy]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None

    def render(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        m = self.margin
        canvas = np.full((h + 2 * m, w + 2 * m, 3), 40, np.uint8)
        canvas[m:m + h, m:m + w] = image

        if self.mask is not None:  # mask 半透明疊色
            overlay = canvas[m:m + h, m:m + w]
            overlay[self.mask] = (overlay[self.mask] * 0.6 + np.array([0, 140, 0]) * 0.4).astype(np.uint8)

        for p in self.pos_points:
            cv2.circle(canvas, self.to_canvas(p), 4, (0, 255, 0), -1)
        for p in self.neg_points:
            cv2.circle(canvas, self.to_canvas(p), 4, (0, 0, 255), -1)

        if self.corners is not None:
            pts = np.array([self.to_canvas(c) for c in self.corners], np.int32)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)
            for i, (pt, color) in enumerate(zip(pts, CORNER_COLORS)):
                filled = -1 if self.visibility[i] else 2  # 不可見的角畫空心
                cv2.circle(canvas, tuple(pt), 8, color, filled)
                cv2.putText(canvas, f"{i + 1}", (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return canvas

    def save(self, frame_path: Path, image_shape) -> Path:
        out = {
            "image": frame_path.name,
            "image_width": image_shape[1],
            "image_height": image_shape[0],
            "corner_order": CORNER_NAMES,
            "corners": {name: [round(float(x), 1), round(float(y), 1)]
                        for name, (x, y) in zip(CORNER_NAMES, self.corners)},
            "visibility": {name: v for name, v in zip(CORNER_NAMES, self.visibility)},
        }
        out_path = ANNOTATIONS_DIR / (frame_path.stem + ".json")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sam2.1_b.pt",
                        help="SAM 權重 (預設 sam2.1_b.pt,首次執行自動下載;想更快可用 sam2.1_t.pt)")
    parser.add_argument("--margin", type=int, default=200, help="畫布留邊像素,讓角點可拖出畫面外")
    args = parser.parse_args()

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    todo = [f for f in frames if not (ANNOTATIONS_DIR / (f.stem + ".json")).exists()]
    print(f"共 {len(frames)} 格,待標註 {len(todo)} 格")
    if not todo:
        return

    annotator = Annotator(args.model, args.margin)
    win = "annotate  [左鍵:桌面/拖角  右鍵:排除  1-4:可見性  s:存  n:跳過  r:重設  q:離開]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, annotator.on_mouse)

    idx = 0
    while idx < len(todo):
        frame_path = todo[idx]
        image = cv2.imread(str(frame_path))
        annotator.reset()
        annotator.pending_sam = False
        print(f"\n[{idx + 1}/{len(todo)}] {frame_path.name}")

        while True:
            if annotator.pending_sam:
                annotator.pending_sam = False
                annotator.run_sam(image)
            cv2.imshow(win, annotator.render(image))
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord("r"):
                annotator.reset()
                annotator.pending_sam = False
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                i = key - ord("1")
                annotator.visibility[i] = not annotator.visibility[i]
                print(f"  {CORNER_LABELS_ZH[i]} 可見性 -> {annotator.visibility[i]}")
            elif key == ord("n"):
                idx += 1
                break
            elif key == ord("s"):
                if annotator.corners is None:
                    print("  尚無角點,先左鍵點桌面")
                    continue
                out_path = annotator.save(frame_path, image.shape)
                print(f"  已存 {out_path.name}")
                idx += 1
                break

    cv2.destroyAllWindows()
    print("全部標註完成")


if __name__ == "__main__":
    main()
