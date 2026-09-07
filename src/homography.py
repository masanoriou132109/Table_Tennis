"""桌面 homography: 影像像素 ↔ 真實桌面座標 (公尺)。

國際標準桌面尺寸: 長 2.74 m × 寬 1.525 m,球網把長邊二等分 (x=1.37)。
角點順序 far_left, far_right, near_right, near_left 對應桌面座標:
    far_left  = (0,    0)
    far_right = (2.74, 0)
    near_right= (2.74, 1.525)
    near_left = (0,    1.525)
(遠側 = 畫面上方那側,見 README 角點約定)

每幀用當幀 4 角重算 homography,故相機 zoom/pan/角度變化皆自動處理。
落點分析: 反彈幀的球影像座標 → image_to_table → 得真實桌面落點。
"""

from __future__ import annotations

import cv2
import numpy as np

TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525
NET_X = TABLE_LENGTH_M / 2      # 球網位置 (長邊中點)
CENTER_Y = TABLE_WIDTH_M / 2    # 雙打中線

# 桌面座標系四角 (公尺),順序同 CORNER_NAMES
TABLE_CORNERS_M = np.array(
    [[0.0, 0.0], [TABLE_LENGTH_M, 0.0],
     [TABLE_LENGTH_M, TABLE_WIDTH_M], [0.0, TABLE_WIDTH_M]], np.float32)


def table_homography(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corners: (4,2) 影像角點 (far_left,far_right,near_right,near_left)。
    回傳 (H_img2table, H_table2img)。"""
    src = corners.astype(np.float32)
    H_img2table = cv2.getPerspectiveTransform(src, TABLE_CORNERS_M)
    H_table2img = cv2.getPerspectiveTransform(TABLE_CORNERS_M, src)
    return H_img2table, H_table2img


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def image_to_table(H_img2table: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """影像像素 → 桌面座標 (公尺)。pts_px: (N,2) 或 (2,)。"""
    return _apply(H_img2table, pts_px)


def table_to_image(H_table2img: np.ndarray, pts_m: np.ndarray) -> np.ndarray:
    """桌面座標 (公尺) → 影像像素。"""
    return _apply(H_table2img, pts_m)


def draw_table_grid(frame: np.ndarray, corners: np.ndarray,
                    grid_m: float = TABLE_WIDTH_M / 4) -> np.ndarray:
    """在畫面上疊出鎖在桌面的真實座標網格 (證明每幀座標對應)。"""
    _, H_t2i = table_homography(corners)

    def line_m(p0, p1, color, thick=1):
        (x0, y0), (x1, y1) = table_to_image(H_t2i, np.array([p0, p1]))
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, thick, cv2.LINE_AA)

    # 細網格 (每 grid_m 公尺)
    x = 0.0
    while x <= TABLE_LENGTH_M + 1e-6:
        line_m((x, 0), (x, TABLE_WIDTH_M), (120, 120, 120), 1)
        x += grid_m
    y = 0.0
    while y <= TABLE_WIDTH_M + 1e-6:
        line_m((0, y), (TABLE_LENGTH_M, y), (120, 120, 120), 1)
        y += grid_m

    line_m((NET_X, 0), (NET_X, TABLE_WIDTH_M), (0, 200, 255), 2)       # 球網 (黃)
    line_m((0, CENTER_Y), (TABLE_LENGTH_M, CENTER_Y), (255, 200, 0), 1)  # 中線 (青)
    # 外框
    for a, b in [((0, 0), (TABLE_LENGTH_M, 0)), ((TABLE_LENGTH_M, 0), (TABLE_LENGTH_M, TABLE_WIDTH_M)),
                 ((TABLE_LENGTH_M, TABLE_WIDTH_M), (0, TABLE_WIDTH_M)), ((0, TABLE_WIDTH_M), (0, 0))]:
        line_m(a, b, (0, 255, 0), 2)
    return frame
