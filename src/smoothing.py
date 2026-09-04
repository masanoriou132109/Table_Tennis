"""桌面 4 角點的時序平滑 (供影片即時推論用)。

策略:
  - presence 低於門檻 → 視為畫面無桌面,重置狀態。
  - 高信心角: 用 EMA 平滑,消除逐格抖動。
  - 低信心角 (被球員遮擋/出畫面): 不信模型的亂猜。改用「上一狀態 → 本格高信心角」
    估計的相似變換 (旋轉+縮放+平移),把上一格該角的位置搬到本格,讓被擋的角
    跟著相機 zoom/pan 移動。至少要 2 個高信心角才能估變換,否則沿用上一位置。

只用 numpy + cv2,無狀態外部依賴。
"""

from __future__ import annotations

import cv2
import numpy as np


class CornerSmoother:
    def __init__(self, alpha: float = 0.5, score_thresh: float = 0.35,
                 presence_thresh: float = 0.5):
        self.alpha = alpha
        self.score_thresh = score_thresh
        self.presence_thresh = presence_thresh
        self.state: np.ndarray | None = None  # (4,2) 平滑後座標
        # 記錄每個角「有信心」的旗標 (回傳供上色/除錯)
        self.confident = np.zeros(4, bool)

    def reset(self) -> None:
        self.state = None
        self.confident = np.zeros(4, bool)

    def update(self, coords: np.ndarray, scores: np.ndarray,
               presence: float) -> np.ndarray | None:
        """吃單格的原始預測,回傳平滑後的 4 角 (或 None 表示無桌面)。"""
        if presence < self.presence_thresh:
            self.reset()
            return None

        conf = scores >= self.score_thresh
        self.confident = conf

        if self.state is None:  # 首格: 直接採用原始預測
            self.state = coords.astype(np.float64).copy()
            return self.state.copy()

        prev = self.state.copy()
        new = self.state.copy()

        # 高信心角: EMA 平滑
        for i in range(4):
            if conf[i]:
                new[i] = (1 - self.alpha) * prev[i] + self.alpha * coords[i]

        # 低信心角: 用高信心角的相似變換從 prev 推算
        occ = np.where(~conf)[0]
        if len(occ) > 0 and conf.sum() >= 2:
            src = prev[conf].astype(np.float32)
            dst = new[conf].astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            if M is not None:
                for i in occ:
                    p = M @ np.array([prev[i][0], prev[i][1], 1.0])
                    new[i] = p
        # (conf<2 時 occ 角沿用 prev 位置,即 new 保持不變)

        self.state = new
        return new.copy()
