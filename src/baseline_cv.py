"""傳統 CV baseline: 以 WTT 桌緣的亮紫粉色 (magenta) 邊線偵測桌面四邊形。

流程:
  1. HSV 閾值取出 magenta 亮線 (閾值由標註沿線取樣統計而來,見 README)
  2. 形態學閉運算把邊線連成封閉的環
  3. 連通元件過濾: 排除碰到畫面邊界的元件 (場地圍欄的同色線通常延伸出畫面)
  4. 取最大合格元件的凸包 -> 擬合四邊形 -> 按語意順序排列角點

已知限制: zoom in 導致桌面貼到畫面邊界時會失敗;球員大面積遮擋邊線時角點會偏。
"""

from __future__ import annotations

import cv2
import numpy as np

# 由 59 張標註沿桌緣取樣 11800 點統計 (p5~p95 範圍稍加放寬)
HSV_LOWER = np.array([125, 30, 120])
HSV_UPPER = np.array([162, 130, 255])

CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]


def fit_quad(mask: np.ndarray) -> np.ndarray | None:
    """從二值 mask 擬合凸四邊形 (與 annotate_sam.py 邏輯一致)。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps_ratio in np.arange(0.01, 0.15, 0.01):
        approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return box.astype(np.float64)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """排成固定語意順序: 遠左, 遠右, 近右, 近左 (遠 = 畫面上方)。"""
    pts = pts[np.argsort(pts[:, 1])]
    far, near = pts[:2], pts[2:]
    far = far[np.argsort(far[:, 0])]
    near = near[np.argsort(near[:, 0])]
    return np.array([far[0], far[1], near[1], near[0]])


def shrink_quad(quad: np.ndarray, offset: float) -> np.ndarray:
    """把四邊形每條邊沿內法線平移 offset 像素,重算交點。
    用途: 凸包貼的是 magenta 線條外緣,向內縮半個線寬對齊桌角。"""
    center = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]])
        n /= np.linalg.norm(n)
        if np.dot(center - a, n) < 0:  # 讓法線指向內部
            n = -n
        a2, b2 = a + n * offset, b + n * offset
        # 齊次座標表示的直線
        lines.append(np.cross([*a2, 1.0], [*b2, 1.0]))
    out = np.empty((4, 2))
    for i in range(4):
        p = np.cross(lines[(i - 1) % 4], lines[i])  # 邊 i-1 與邊 i 的交點 = 角 i
        out[i] = p[:2] / p[2]
    return out


def detect_table_quad(image: np.ndarray, border_margin: int = 5,
                      min_area_ratio: float = 0.02, max_area_ratio: float = 0.6,
                      edge_inward_offset: float = 0.0) -> np.ndarray | None:
    """偵測桌面四邊形,回傳 4x2 角點 (遠左, 遠右, 近右, 近左);失敗回傳 None。"""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # 閉運算連接被球網/球員切斷的邊線
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    candidates = []
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        # 碰到畫面邊界的元件多半是場地圍欄線,排除
        if x < border_margin or y < border_margin or \
           x + cw > w - border_margin or y + ch > h - border_margin:
            continue
        if area < 500:  # 雜點
            continue
        candidates.append((area, i))

    for _, comp_id in sorted(candidates, reverse=True):
        comp_mask = (labels == comp_id).astype(np.uint8)
        quad = fit_quad(comp_mask)
        if quad is None:
            continue
        quad_area = cv2.contourArea(quad.astype(np.float32))
        if min_area_ratio * w * h <= quad_area <= max_area_ratio * w * h:
            return order_corners(shrink_quad(quad, edge_inward_offset))
    return None
