"""桌面 4 角點的時序平滑 (供影片即時推論用)。

策略:
  - presence 低於門檻 → 視為畫面無桌面,重置狀態。
  - 高信心角: 用 EMA 平滑,消除逐格抖動。角一恢復可見就拉回模型預測,不長期漂移。
  - 低信心角 (被球員遮擋/出畫面): 不信模型的亂猜。用「上一狀態的可見角 → 本格可見角」
    估計的平面變換,把上一格該角的位置搬到本格,讓被擋的角跟著相機 zoom/pan 移動。
    依可見角數量選變換: 3 角→完整仿射 (6 DOF, 含剪切), 2 角→相似 (4 DOF), <2→沿用。
    再依相機運動量在「沿用」與「變換」間混合 (hybrid): 相機靜止時偏沿用 (避免變換把
    可見角抖動經槓桿放大到遠處被擋角),運鏡時偏變換 (追隨相機,避免長遮擋凍結漂移)。

只用 numpy + cv2,場地無關 (只吃角點座標,不吃顏色),可隨模型一起部署。
合成遮擋測試 (見 scripts/eval_occlusion_fill.py): hybrid 在靜止/運鏡兩情境皆穩健,
優於純相似 (最差) 與純沿用 (運鏡時長遮擋會漂)。
"""

from __future__ import annotations

import cv2
import numpy as np


class CornerSmoother:
    def __init__(self, alpha: float = 0.5, score_thresh: float = 0.35,
                 presence_thresh: float = 0.5, motion_ref: float = 3.0):
        self.alpha = alpha
        self.score_thresh = score_thresh
        self.presence_thresh = presence_thresh
        self.motion_ref = motion_ref  # px: 可見角平均位移達此值即完全採用變換
        self.state: np.ndarray | None = None  # (4,2) 平滑後座標
        # 記錄每個角「有信心」的旗標 (回傳供上色/除錯)
        self.confident = np.zeros(4, bool)
        # 每個角是否「曾被可信偵測過」: 未曾見過的角無法憑幾何補出 (透視), 不可信任
        self.seen = np.zeros(4, bool)

    def reset(self) -> None:
        self.state = None
        self.confident = np.zeros(4, bool)
        self.seen = np.zeros(4, bool)

    @property
    def all_seen(self) -> bool:
        """4 角是否都至少被可信偵測過一次 (建立透視參考的前提)。"""
        return bool(self.seen.all())

    def update(self, coords: np.ndarray, scores: np.ndarray,
               presence: float) -> np.ndarray | None:
        """吃單格的原始預測,回傳平滑後的 4 角 (或 None 表示無桌面)。"""
        if presence < self.presence_thresh:
            self.reset()
            return None

        conf = scores >= self.score_thresh
        self.confident = conf

        if self.state is None:  # 首格: 建立佔位狀態 (未見過的角為垃圾, 靠 seen 標記)
            self.state = coords.astype(np.float64).copy()
            self.seen = conf.copy()
            return self.state.copy()

        prev = self.state.copy()
        new = self.state.copy()

        # 高信心角: 首次可見 → 直接採用 (prev 為佔位垃圾, 不可 EMA); 已見過 → EMA 平滑
        for i in range(4):
            if conf[i]:
                new[i] = coords[i] if not self.seen[i] else \
                    (1 - self.alpha) * prev[i] + self.alpha * coords[i]
        self.seen |= conf

        # 低信心角: 用可見角估平面變換,依相機運動量在「沿用」與「變換」間混合
        occ = np.where(~conf)[0]
        n_conf = int(conf.sum())
        if len(occ) > 0 and n_conf >= 2:
            src = prev[conf].astype(np.float32)
            dst = new[conf].astype(np.float32)
            if n_conf >= 3:  # 完整仿射 (6 DOF): 平移+旋轉+縮放+剪切
                M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
            else:            # 相似 (4 DOF): 平移+旋轉+等比縮放
                M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
            motion = float(np.linalg.norm(dst - src, axis=1).mean())
            w = min(motion / self.motion_ref, 1.0)  # 靜止→沿用, 運鏡→變換
            for i in occ:
                hold = prev[i]
                if M is not None:
                    warped = M @ np.array([prev[i][0], prev[i][1], 1.0])
                    new[i] = (1 - w) * hold + w * warped
                else:
                    new[i] = hold
        # (conf<2 時 occ 角沿用 prev 位置,即 new 保持不變)

        self.state = new
        return new.copy()
