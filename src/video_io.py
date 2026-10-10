"""處理解析度與 demo 輸出解析度。

所有像素單位的門檻 (edge 精修、追蹤器、教授的去抖動/跳動門檻) 都是在 1280x720 上
校準並經肉眼驗證的。因此不論來源影片多大, 一律在記憶體內縮成 PROC 尺寸再處理,
門檻就不必跟著解析度改。球模型 (704x704) 與桌面模型 (512x288) 本來就會縮小輸入,
用原圖或 720p 對它們沒有差別。

demo 一律輸出 OUT 尺寸 (1080p), 從原始畫面縮放而來 (保留細節, 不是 720p 放大);
疊圖座標與線寬/字體大小乘以 UI 倍率。
"""

from __future__ import annotations

import cv2
import numpy as np

PROC_W, PROC_H = 1280, 720
OUT_W, OUT_H = 1920, 1080
UI = OUT_H / PROC_H   # 處理座標 → demo 座標的倍率 (1.5)


def to_proc(frame: np.ndarray) -> np.ndarray:
    """原始畫面 → 處理尺寸 (1280x720)。"""
    h, w = frame.shape[:2]
    if (w, h) == (PROC_W, PROC_H):
        return frame
    return cv2.resize(frame, (PROC_W, PROC_H),
                      interpolation=cv2.INTER_AREA if w > PROC_W else cv2.INTER_LINEAR)


def to_out(frame: np.ndarray) -> np.ndarray:
    """原始畫面 → demo 輸出尺寸 (1920x1080)。"""
    h, w = frame.shape[:2]
    if (w, h) == (OUT_W, OUT_H):
        return frame.copy()
    return cv2.resize(frame, (OUT_W, OUT_H),
                      interpolation=cv2.INTER_AREA if w > OUT_W else cv2.INTER_LINEAR)


def check_aspect(w: int, h: int) -> None:
    """非 16:9 會被拉伸變形 (桌面比例與 homography 會錯), 直接報錯而不是悄悄算錯。"""
    if abs(w / h - PROC_W / PROC_H) > 0.01:
        raise SystemExit(f"影片 {w}x{h} 不是 16:9, 目前的處理流程只支援 16:9")


def s(v: float) -> int:
    """長度 (處理座標或 720p 下設計的 UI 尺寸) → demo 像素。"""
    return int(round(v * UI))


def pt(x: float, y: float) -> tuple[int, int]:
    return s(x), s(y)


def fs(scale: float) -> float:
    """字體大小。"""
    return scale * UI


def lw(t: int) -> int:
    """線寬 (-1 = 填滿, 維持不變)。"""
    return t if t < 0 else max(1, int(round(t * UI)))
