"""桌面四邊形的幾何合理性閘門。

用途: 側拍/斜角重播鏡頭或誤偵時,模型會輸出扭曲或不合理的四邊形。
主鏡位的桌面四邊形有穩定的幾何特性 (由 216 張標註校準):
  - 凸四邊形、不自交叉
  - 順序合理: 遠側兩角在近側兩角上方,左角在右角左側
  - 面積占畫面 2.0~7.3% (放寬到 1.2~15%)
  - 長寬比 (寬/高) 2.19~4.31 (放寬到 1.6~5.5)

任一條件不符 → 視為不可靠,呼叫端應拒絕繪製 (而非畫出垃圾框)。
"""

from __future__ import annotations

import cv2
import numpy as np

# 角點順序: far_left, far_right, near_right, near_left
AREA_FRAC_MIN = 0.012
AREA_FRAC_MAX = 0.15
ASPECT_MIN = 1.6
ASPECT_MAX = 5.5


def is_plausible_quad(quad: np.ndarray, img_w: int, img_h: int) -> bool:
    """quad: (4,2) 依 far_left,far_right,near_right,near_left 順序。"""
    if quad is None or quad.shape != (4, 2) or not np.isfinite(quad).all():
        return False
    pts = quad.astype(np.float64)
    fl, fr, nr, nl = pts

    # 凸且不自交叉 (bowtie 會是非凸)
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False

    # 順序: 遠在近上方, 左在右左側
    if not (fl[1] < nl[1] and fr[1] < nr[1]):
        return False
    if not (fl[0] < fr[0] and nl[0] < nr[0]):
        return False

    # 面積占比
    area = cv2.contourArea(pts.astype(np.float32))
    frac = area / (img_w * img_h)
    if not (AREA_FRAC_MIN <= frac <= AREA_FRAC_MAX):
        return False

    # 長寬比 (上下緣平均寬 / 左右緣平均高)
    width = (np.linalg.norm(fr - fl) + np.linalg.norm(nr - nl)) / 2
    height = (np.linalg.norm(nl - fl) + np.linalg.norm(nr - fr)) / 2
    if height <= 1e-6:
        return False
    aspect = width / height
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False

    return True
