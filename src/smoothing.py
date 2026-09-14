"""桌面 4 角點的時序平滑 + 持久先驗補角 (供影片即時推論用)。

策略:
  - presence 低於門檻 → 視為畫面無桌面,重置狀態。
  - 高信心角: 首次可見直接採用,之後 EMA 平滑消抖動。
  - 持久先驗 (reference): 每當 4 角全可信,記下該完整四邊形當作桌面模板。
    因主轉播相機機位固定,此模板在整場有效 (角位置 std~10px)。
  - 低信心角 (被球員遮擋,常見於發球): 用「先驗可見角 → 本格可見角」的仿射對齊,
    把先驗中該角的位置映射到本格。先驗本身含正確透視,故補出的角透視正確
    (合成遮擋測試 ~5px,遠優於單幀平行四邊形補全 ~49px)。至少 2 個可見角才能對齊。

只要整場曾完整看過桌子一次,之後每個回合開頭即使發球遮角也能立刻畫框
(滿足「一開始就有框」)。只用 numpy + cv2,場地無關,可移植 Swift 供 Core ML。
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
        self.state: np.ndarray | None = None      # (4,2) 平滑後座標
        self.reference: np.ndarray | None = None   # (4,2) 最近一次 4 角全可信的完整四邊形
        self.confident = np.zeros(4, bool)         # 本格各角是否高信心 (供上色)
        self.valid = False                         # 本格輸出是否可靠 (供 tracker 判斷)

    def reset(self) -> None:
        """整段追蹤重置。注意: reference 不清除 (主相機機位固定,模板跨回合有效)。"""
        self.state = None
        self.confident = np.zeros(4, bool)
        self.valid = False

    def _fill_affine(self, dst_conf: np.ndarray, conf: np.ndarray) -> np.ndarray | None:
        """用先驗可見角 → 本格可見角的仿射,回傳映射用的 M (2x3);不足則 None。"""
        n = int(conf.sum())
        src = self.reference[conf].astype(np.float32)
        dst = dst_conf.astype(np.float32)
        if n >= 3:
            M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        else:  # n == 2: 相似變換
            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return M

    def update(self, coords: np.ndarray, scores: np.ndarray,
               presence: float) -> np.ndarray | None:
        """吃單格原始預測,回傳平滑後 4 角 (或 None 表示不可靠/無桌面)。"""
        if presence < self.presence_thresh:
            self.reset()
            return None

        conf = scores >= self.score_thresh
        self.confident = conf
        n_conf = int(conf.sum())

        if n_conf == 4:  # 更新持久先驗為最新完整偵測
            self.reference = coords.astype(np.float64).copy()

        # 平滑高信心角 (首格或首次可見 → 直接採用)
        new = coords.astype(np.float64).copy()
        if self.state is not None:
            for i in range(4):
                if conf[i]:
                    new[i] = (1 - self.alpha) * self.state[i] + self.alpha * coords[i]

        # 補低信心角: 先驗 + 可見角仿射對齊
        occ = ~conf
        if occ.any():
            if self.reference is not None and n_conf >= 2:
                M = self._fill_affine(new[conf], conf)
                if M is not None:
                    for i in np.where(occ)[0]:
                        new[i] = M @ np.array([self.reference[i][0], self.reference[i][1], 1.0])
                    self.valid = True
                elif self.state is not None:      # 仿射失敗 → 沿用上一位置
                    new[occ] = self.state[occ]
                    self.valid = False
                else:
                    self.valid = False
            else:  # 無先驗或可見角不足 → 無法可靠補出被遮角
                if self.state is not None:
                    new[occ] = self.state[occ]
                self.valid = False
        else:
            self.valid = True  # 4 角全可信

        self.state = new
        return new.copy()
