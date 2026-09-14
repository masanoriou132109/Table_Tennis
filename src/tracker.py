"""影片級桌面追蹤: 整合平滑 + 三重穩健性閘門。

單幀模型預測有三類錯誤,各由一道閘門處理:
  1. 幾何不合理 (自交叉/比例異常四邊形) → is_plausible_quad (src/geometry.py)
  2. 全角低信心 (廣角/斜角/分布外鏡頭,模型整體不確定) → 信心閘門 (可信角數 / 平均分數)
  3. 瞬態跳變 / 場景切換殘影 → 時序確認 (連續 N 幀一致才輸出,大跳變重啟追蹤)

輸出經 CornerSmoother 平滑 (含遮擋角 hybrid 補算)。此類別是純幾何+狀態機,
不依賴顏色,設計上可直接移植到 Swift 供 Core ML 部署。

用法:
    tk = TableTracker(img_w, img_h)
    quad = tk.update(coords, scores, presence)  # quad 為 (4,2) 或 None (不繪製)
"""

from __future__ import annotations

import numpy as np

from src.geometry import is_plausible_quad
from src.smoothing import CornerSmoother


class TableTracker:
    def __init__(self, img_w: int, img_h: int, *,
                 score_thresh: float = 0.75, presence_thresh: float = 0.5,
                 min_conf_corners: int = 2, confirm_frames: int = 3,
                 max_misses: int = 5, jump_thresh: float = 100.0):
        self.img_w = img_w
        self.img_h = img_h
        self.score_thresh = score_thresh
        self.presence_thresh = presence_thresh
        self.min_conf_corners = min_conf_corners
        self.confirm_frames = confirm_frames
        self.max_misses = max_misses
        self.jump_thresh = jump_thresh

        self.smoother = CornerSmoother(score_thresh=score_thresh,
                                       presence_thresh=presence_thresh)
        self.last: np.ndarray | None = None   # 最近一次接受的平滑四邊形
        self.hits = 0                          # 連續一致幀數
        self.misses = 0                        # 連續不合格幀數
        self.confirmed = False

    def _reset_track(self) -> None:
        self.last = None
        self.hits = 0
        self.confirmed = False
        self.smoother.reset()

    def update(self, coords: np.ndarray, scores: np.ndarray,
               presence: float) -> np.ndarray | None:
        n_conf = int((scores >= self.score_thresh).sum())
        # 閘門 1+2: presence + 信心 (可信角數不足視為不可靠)
        frame_ok = presence >= self.presence_thresh and n_conf >= self.min_conf_corners

        smoothed = self.smoother.update(coords, scores, presence) if frame_ok else None
        # smoother.valid: 4 角全可信,或有先驗且可用可見角補出被遮角
        if smoothed is not None and not self.smoother.valid:
            frame_ok = False
            smoothed = None
        # 閘門 3-幾何: 平滑後四邊形須合理
        if smoothed is not None and not is_plausible_quad(smoothed, self.img_w, self.img_h):
            frame_ok = False
            smoothed = None

        if not frame_ok:
            self.misses += 1
            self.hits = 0
            if self.misses > self.max_misses:
                self._reset_track()
            self.confirmed = False
            return None

        # 確認狀態機
        if self.last is None or \
           float(np.linalg.norm(smoothed - self.last, axis=1).mean()) <= self.jump_thresh:
            self.hits += 1            # 與既有追蹤一致 → 累積確認
        else:
            self.hits = 1             # 大跳變 (場景切換) → 於新位置重啟 tentative
        self.misses = 0
        self.last = smoothed
        self.confirmed = self.hits >= self.confirm_frames
        return self.last if self.confirmed else None
