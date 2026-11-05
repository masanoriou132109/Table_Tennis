"""離線 (前後看) 的桌面四角平滑: 給錄好的影片用 (demo / 落點分析), 不是即時。

TableTracker 只能看過去的幀, 角被擋住時只能推算 (或信任被誤認的角), 畫面上會偏。
錄好的影片前後都看得到: 被擋的時間 88% <= 0.5s、99% <= 2s, 被擋的角在前後一秒內
幾乎都有可信的觀測。所以每個角各自:
  1. 只用「可信」的觀測: 模型原始信心 >= trust_score (0.9)。3 角吻合時正確的第 4 角
     信心 p5 0.88 / 中位 0.95, 被擋或被誤認 (例如球員衣服) 中位 0.44、>= 0.9 只有 1%。
  2. 每幀取前後 window 秒內的可信觀測, 做加權 (tricube) 局部直線 x(t), y(t) 擬合,
     再用 bisquare 重新加權去掉離群點 → 鏡頭不動時近乎固定, 運鏡時跟著動, 沒有模式切換。
  3. window 內觀測不夠 → 用前後最近的可信觀測 (max_gap 秒內) 線性內插; 只有一側 → 沿用那一側。
     都沒有 → 用追蹤器原本的輸出。
鏡頭切換 (追蹤器沒框的間隔 > cut_gap 秒, 或框突然跳動 > cut_jump px) 切成不同段, 各段分開擬合。

只決定「框的位置」; 哪些幀有框仍由 TableTracker 決定 (presence / 幾何閘門 / 確認)。
"""

from __future__ import annotations

import numpy as np


def _segments(fis: list[int], quads: dict[int, np.ndarray], fps: float,
              cut_gap: float, cut_jump: float) -> list[list[int]]:
    segs: list[list[int]] = []
    for fi in fis:
        if segs:
            prev = segs[-1][-1]
            jump = float(np.linalg.norm(quads[fi] - quads[prev], axis=1).mean())
            if (fi - prev) / fps <= cut_gap and jump <= cut_jump:
                segs[-1].append(fi)
                continue
        segs.append([fi])
    return segs


def _local_fit(ts: np.ndarray, ps: np.ndarray, t: float, window: float, iters: int = 2) -> np.ndarray:
    """t 附近的加權局部直線擬合, bisquare 穩健重新加權。ts: (n,), ps: (n,2)。"""
    u = np.abs(ts - t) / window
    w = np.clip(1 - u ** 3, 0, None) ** 3            # tricube
    X = np.stack([np.ones_like(ts), ts - t], axis=1)
    robust = np.ones_like(ts)
    for it in range(iters + 1):
        ww = w * robust
        A = X.T @ (X * ww[:, None])
        if np.linalg.cond(A) > 1e8:                    # 觀測集中在一點 → 退化成加權平均
            return (ps * ww[:, None]).sum(0) / max(ww.sum(), 1e-9)
        beta = np.linalg.solve(A, X.T @ (ps * ww[:, None]))
        if it == iters:
            return beta[0]
        res = np.linalg.norm(ps - X @ beta, axis=1)
        s = np.median(res[ww > 0]) if (ww > 0).any() else 0.0
        if s < 1e-6:
            return beta[0]
        robust = np.clip(1 - (res / (6 * s)) ** 2, 0, None) ** 2   # bisquare
    return beta[0]


def _similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """最小平方相似變換 (旋轉+等比縮放+平移): 回傳 (A 2x2, b), dst ≈ src @ A.T + b。"""
    if len(src) < 2:
        return None
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    den = (s0 ** 2).sum()
    if den < 1e-9:
        return None
    a = (s0 * d0).sum() / den
    b = (s0[:, 0] * d0[:, 1] - s0[:, 1] * d0[:, 0]).sum() / den
    A = np.array([[a, -b], [b, a]])
    return A, md - ms @ A.T


def smooth_quads(frames: list[int], coords: np.ndarray, scores: np.ndarray,
                 tracker_quads: dict[int, np.ndarray], fps: float, *,
                 trust_score: float = 0.9, window: float = 0.5, min_obs: int = 6,
                 max_gap: float = 2.0, cut_gap: float = 0.5, cut_jump: float = 40.0) -> dict[int, np.ndarray]:
    """
    frames: 每幀的幀號 (依序); coords (n,4,2) / scores (n,4): 桌面模型原始輸出 (處理座標)
    tracker_quads: 幀 → TableTracker 輸出的框 (沒框的幀不放)
    回傳: 幀 → 平滑後的框 (只含 tracker_quads 有的幀)
    """
    idx = {fi: k for k, fi in enumerate(frames)}
    out: dict[int, np.ndarray] = {}
    for seg in _segments(sorted(tracker_quads), tracker_quads, fps, cut_gap, cut_jump):
        ks = np.array([idx[fi] for fi in seg])
        ts = np.array(seg, float) / fps
        n = len(seg)
        Q = np.stack([tracker_quads[fi].astype(float) for fi in seg])     # 預設: 追蹤器的
        ok = np.zeros((n, 4), bool)                                       # 該幀該角有「直接」估計

        # 第一輪: 前後 window 秒內可信觀測夠多 (而且兩側都有) → 局部直線擬合
        for c in range(4):
            good = scores[ks, c] >= trust_score
            gt, gp = ts[good], coords[ks[good], c].astype(float)
            if len(gt) == 0:
                continue                                                  # 這段這個角從沒可信過
            for j in range(n):
                near = np.abs(gt - ts[j]) <= window
                if near.sum() >= min_obs and (gt[near] <= ts[j]).any() and (gt[near] >= ts[j]).any():
                    Q[j, c] = _local_fit(gt[near], gp[near], ts[j], window)
                    ok[j, c] = True

        # 第二輪: 被擋的區間, 讓被擋的角跟著「看得到的角」一起動 (鏡頭在動時直線內插跟不上):
        # 以區間前、後最近的直接估計幀為錨點, 用兩幀都有直接估計的其他角 (>= 2 個) 算相似變換,
        # 套到錨點上這個角的位置; 前後兩個預測依時間加權。共同的角不夠 → 錨點之間直線內插。
        R = Q.copy()
        for c in range(4):
            okc = np.where(ok[:, c])[0]
            if len(okc) == 0:
                continue
            for j in np.where(~ok[:, c])[0]:
                pos = np.searchsorted(okc, j)
                anchors = []
                if pos > 0 and ts[j] - ts[okc[pos - 1]] <= max_gap:
                    anchors.append(okc[pos - 1])
                if pos < len(okc) and ts[okc[pos]] - ts[j] <= max_gap:
                    anchors.append(okc[pos])
                if not anchors:
                    continue
                preds, wts = [], []
                for a in anchors:
                    common = [i for i in range(4) if i != c and ok[a, i] and ok[j, i]]
                    M = _similarity(Q[a, common], Q[j, common]) if len(common) >= 2 else None
                    preds.append(Q[a, c] @ M[0].T + M[1] if M is not None else Q[a, c])
                    wts.append(1.0 / (abs(ts[j] - ts[a]) + 1e-3))
                R[j, c] = np.average(np.stack(preds), axis=0, weights=wts)
        for j, fi in enumerate(seg):
            out[fi] = R[j]
    return out
