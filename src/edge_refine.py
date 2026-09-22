"""用桌面邊線把四邊形的角點精修 (亮度梯度 + RANSAC,跨場地通用)。

想法: 角點可能被球員遮住,但通往它的兩條邊線通常還看得見一段 —
「一條線即使失去一段仍然穩健」。沿預期邊線位置局部搜尋最強亮度梯度,
從中找出共線的共識集,擬合直線,相鄰兩線求交點得到次像素角點。

兩道關鍵過濾 (診斷 data/diag_edge 的邊帶圖後加入):
  1. 梯度極性: 桌面邊界整條的明暗方向一致;桌下 LED 圖騰、球員邊緣的極性反覆。
     先取多數極性,反向的點直接丟棄。
  2. RANSAC: 污染是「結構性」的 (例如紅點會忠實描出 LED 圖騰的弧線),
     Huber 只降權重仍會被拉走;RANSAC 取最大共識集,才能整批拋棄。

接受條件 (不滿足就不修該角,寧可不動也不亂動):
  內點數、內點跨距,以及由此推估的外插誤差 2σD/(S·√N) 必須夠小。
"""

from __future__ import annotations

import cv2
import numpy as np

SEARCH = 9            # px: 沿法線方向的搜尋半徑
N_SAMPLES = 40        # 每條邊取樣點數 (較密有助 RANSAC 找共識)
EDGE_MARGIN = 0.10    # 邊的兩端各跳過此比例
MIN_GRAD = 12.0       # 梯度強度下限

RANSAC_TOL = 1.5      # px: 內點的垂直距離容差
RANSAC_ITERS = 200
# 下面兩個只是防退化的下限;真正的判準是 MAX_PRED_ERR (誤差公式已含跨距/點數/外插距離)
MIN_INLIERS = 8       # 連續段的點數下限
MIN_SPAN_PX = 20.0    # px: 連續段的跨距下限
MAX_RUN_GAP = 2       # 連續段容許中斷幾個取樣點 (球衣白字等孤立群會被切開)
MAX_PRED_ERR = 3.0    # px: 外插到角點的推估誤差上限 2σD/(S·√N)
NOISE_PX = 0.7        # px: 假設的每點垂直雜訊 (用於誤差推估)
MAX_ANGLE_DEV = 12.0  # deg: 擬合線與參考邊方向的最大夾角 (擋掉選錯連續段)


def _sample_edge(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                 t_lo: float, t_hi: float):
    """沿 p0→p1 的 [t_lo,t_hi] 取樣,每點沿法線找最強梯度。

    回傳 (pts, signed_grad, ts, length):
      pts         (N,2) 找到的邊點影像座標
      signed_grad (N,)  該點梯度的正負號 (供極性過濾)
      ts          (N,)  沿邊的參數位置 0~1
      idx         (N,)  原始取樣序號 (供「連續段」判定)
    """
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if length < 20:
        return np.empty((0, 2)), np.empty(0), np.empty(0), np.empty(0, int), length
    d = d / length
    n = np.array([-d[1], d[0]])
    offs = np.arange(-SEARCH, SEARCH + 1, dtype=np.float64)
    h, w = gray.shape

    pts, sg, ts, idx = [], [], [], []
    for j, t in enumerate(np.linspace(t_lo, t_hi, N_SAMPLES)):
        base = p0 + d * (t * length)
        line = base[None, :] + n[None, :] * offs[:, None]
        xs = line[:, 0].astype(np.float32).reshape(-1, 1)
        ys = line[:, 1].astype(np.float32).reshape(-1, 1)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > w - 2 or ys.max() > h - 2:
            continue
        prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR).ravel()
        diff = np.diff(prof)
        k = int(np.argmax(np.abs(diff)))
        if abs(diff[k]) < MIN_GRAD:
            continue
        # 鄰近三點拋物線內插取次像素峰位
        g = np.abs(diff)
        if 0 < k < len(g) - 1:
            a, b, c = float(g[k - 1]), float(g[k]), float(g[k + 1])
            den = a - 2 * b + c
            sub = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
        else:
            sub = 0.0
        off = offs[k] + 0.5 + float(np.clip(sub, -1, 1))
        pts.append(base + n * off)
        sg.append(np.sign(diff[k]))
        ts.append(t)
        idx.append(j)
    return (np.array(pts) if pts else np.empty((0, 2)),
            np.array(sg), np.array(ts), np.array(idx, int), length)


def _polarity_filter(pts, sg, ts, idx):
    """只保留多數梯度極性的點 (桌邊極性一致, LED/人體邊緣反覆)。"""
    if len(pts) == 0:
        return pts, ts, idx, np.zeros(0, bool)
    keep = sg == (1.0 if (sg > 0).sum() >= (sg < 0).sum() else -1.0)
    return pts[keep], ts[keep], idx[keep], keep


def _longest_run(idx: np.ndarray) -> np.ndarray:
    """在取樣序號中找最長的連續段 (容許中斷 MAX_RUN_GAP 個)。回傳布林遮罩。

    桌面邊界會產生一長串連續的點;球衣白字、場地標記等只會產生孤立小群,
    會被中斷處切開而落選 —— 這是「共線」之外必要的第二個條件。
    """
    n = len(idx)
    if n == 0:
        return np.zeros(0, bool)
    starts = [0]
    for k in range(1, n):
        if idx[k] - idx[k - 1] > MAX_RUN_GAP + 1:
            starts.append(k)
    starts.append(n)
    best = max(zip(starts[:-1], starts[1:]), key=lambda ab: ab[1] - ab[0])
    mask = np.zeros(n, bool)
    mask[best[0]:best[1]] = True
    return mask


def _ransac_line(pts: np.ndarray, rng: np.random.Generator):
    """回傳 (line(a,b,c), inlier_mask);找不到足夠共識回傳 (None, None)。"""
    n = len(pts)
    if n < MIN_INLIERS:
        return None, None
    best_mask, best_cnt = None, 0
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(n, 2, replace=False)
        v = pts[j] - pts[i]
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        nrm = np.array([-v[1], v[0]]) / L
        dist = np.abs((pts - pts[i]) @ nrm)
        mask = dist <= RANSAC_TOL
        cnt = int(mask.sum())
        if cnt > best_cnt:
            best_mask, best_cnt = mask, cnt
    if best_mask is None or best_cnt < MIN_INLIERS:
        return None, None
    # 只用內點重新擬合 (最小平方)
    inl = pts[best_mask]
    vx, vy, x0, y0 = cv2.fitLine(inl.astype(np.float32), cv2.DIST_L2,
                                 0, 0.01, 0.01).ravel()
    return np.array([vy, -vx, vx * y0 - vy * x0], np.float64), best_mask


def fit_edge(gray, p0, p1, t_lo, t_hi, rng):
    """擬合一條邊: 取樣 → 極性過濾 → RANSAC 共線 → 取最長連續段 → 重新擬合。

    回傳 dict 含 line / 內點遮罩 / 跨距等,供精修與診斷共用。
    """
    pts, sg, ts, idx, length = _sample_edge(gray, p0, p1, t_lo, t_hi)
    kept_pts, kept_ts, kept_idx, pol_mask = _polarity_filter(pts, sg, ts, idx)
    info = {"all_pts": pts, "pol_mask": pol_mask, "kept_pts": kept_pts,
            "line": None, "inlier_mask": None, "length": length,
            "span": 0.0, "n_inliers": 0, "centroid": None, "n_ransac": 0}
    if len(kept_pts) == 0:
        return info
    line, inl = _ransac_line(kept_pts, rng)
    if line is None:
        return info
    info["n_ransac"] = int(inl.sum())
    # 只留最長連續段, 切掉球衣白字之類的孤立群, 再用它重新擬合
    run = np.zeros(len(kept_pts), bool)
    sub = _longest_run(kept_idx[inl])
    run[np.where(inl)[0][sub]] = True
    if run.sum() < 2:
        return info
    final = kept_pts[run]
    vx, vy, x0, y0 = cv2.fitLine(final.astype(np.float32), cv2.DIST_L2,
                                 0, 0.01, 0.01).ravel()
    info["line"] = np.array([vy, -vx, vx * y0 - vy * x0], np.float64)
    info["inlier_mask"] = run
    it = kept_ts[run]
    info["span"] = float(it.max() - it.min()) * length
    info["n_inliers"] = int(run.sum())
    info["centroid"] = final.mean(axis=0)
    return info


def _pred_error(info, corner: np.ndarray) -> float:
    """外插到 corner 的推估誤差 2σD/(S·√N)。"""
    if info["centroid"] is None or info["span"] <= 0:
        return float("inf")
    D = float(np.linalg.norm(corner - info["centroid"]))
    return 2 * NOISE_PX * D / (info["span"] * np.sqrt(info["n_inliers"]))


def _angle_dev(line, ref_dir) -> float:
    """擬合線方向與參考方向的夾角 (度, 0~90)。"""
    a, b, _ = line
    d = np.array([-b, a])                 # 線方向 = 法線轉 90 度
    n1 = np.linalg.norm(d); n2 = np.linalg.norm(ref_dir)
    if n1 < 1e-9 or n2 < 1e-9:
        return 90.0
    cos = abs(float(d @ ref_dir) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, 0, 1))))


def edge_accepted(info, ref_dir=None) -> bool:
    """跨距/點數只是防退化下限;主判準是誤差公式與角度一致性。

    ref_dir: 參考邊方向 (優先用追蹤器的持久先驗 — 那是四角全可見時記下的,
    方向可信;而當前 quad 的邊若有一端被補錯, 方向本身就是歪的, 不能當基準)。
    """
    if info["line"] is None or info["n_inliers"] < MIN_INLIERS \
            or info["span"] < MIN_SPAN_PX:
        return False
    if ref_dir is not None and _angle_dev(info["line"], ref_dir) > MAX_ANGLE_DEV:
        return False
    return True


def refine_quad(image: np.ndarray, quad: np.ndarray,
                confident: np.ndarray | None = None,
                reference: np.ndarray | None = None,
                shift_confident: float = 4.0,
                shift_occluded: float = 45.0,
                seed: int = 0):
    """用邊線精修四邊形角點。

    confident: (4,) bool — 哪些角是模型高信心給出的。容許位移依信心分開設定:
      可信角只微調 (模型已準到約 2px);被遮角由先驗補出可能偏 15~40px,
      此時邊線交點是唯一的真實影像證據,允許大幅修正。

    回傳 (refined_quad, ok, infos) — ok[i] 表示該角確實被更新;
    infos 為 4 條邊的擬合資訊 (供診斷)。
    """
    if confident is None:
        confident = np.ones(4, bool)
    rng = np.random.default_rng(seed)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 必須轉 float: uint8 上做差分會溢位繞回
    gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)

    # 整條邊都取樣。曾經為了避開遮擋端而只取中段, 但那是在還沒有極性過濾/RANSAC/
    # 最長連續段之前的設計 —— 現在那三道過濾本來就在處理遮擋物, 限制範圍反而
    # 把好樣本一起丟掉 (兩端都被遮時只剩 30% 邊長, 點數與跨距都不足而整條被拒)。
    infos = []
    for i in range(4):
        a, b = i, (i + 1) % 4
        infos.append(fit_edge(gray, quad[a].astype(np.float64), quad[b].astype(np.float64),
                              EDGE_MARGIN, 1 - EDGE_MARGIN, rng))

    base = reference if reference is not None else quad
    ref_dirs = [base[(i + 1) % 4] - base[i] for i in range(4)]
    for i in range(4):
        infos[i]["angle_dev"] = (_angle_dev(infos[i]["line"], ref_dirs[i])
                                 if infos[i]["line"] is not None else float("nan"))
        infos[i]["accepted"] = edge_accepted(infos[i], ref_dirs[i])

    out = quad.astype(np.float64).copy()
    ok = np.zeros(4, bool)
    for i in range(4):
        i1, i2 = (i - 1) % 4, i          # 角 i = 邊 i-1 與 邊 i 的交點
        e1, e2 = infos[i1], infos[i2]
        if not (e1["accepted"] and e2["accepted"]):
            continue
        p = np.cross(e1["line"], e2["line"])
        if abs(p[2]) < 1e-9:
            continue
        cand = p[:2] / p[2]
        # 外插誤差推估: 兩條邊都要夠可靠
        if max(_pred_error(e1, cand), _pred_error(e2, cand)) > MAX_PRED_ERR:
            continue
        limit = shift_confident if confident[i] else shift_occluded
        if np.linalg.norm(cand - quad[i]) <= limit:
            out[i] = cand
            ok[i] = True
    return out, ok, infos
