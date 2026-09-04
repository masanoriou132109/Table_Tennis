"""桌面角點資料集: 59 張標註 (正樣本) + 27 張無桌面特寫 (負樣本)。

訓練時的資料增強 (全部以一個 3x3 homography 合成,同步變換角點座標):
  - 隨機縮放 0.7~1.6 倍 (模擬 zoom in/out) + 隨機平移
  - 小幅透視擾動 (模擬鏡頭角度變化)
  - 水平翻轉 (交換 左右 角點語意)
  - HSV 色相/飽和/亮度抖動 (模擬不同場地燈光配色)

角點經增強後跑出畫面 -> 該角 heatmap 為零、visibility 目標為 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.model import HEATMAP_STRIDE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

CORNER_NAMES = ["far_left", "far_right", "near_right", "near_left"]
FLIP_PERM = [1, 0, 3, 2]  # 水平翻轉後 遠左<->遠右, 近右<->近左

INPUT_W, INPUT_H = 512, 288
HM_W, HM_H = INPUT_W // HEATMAP_STRIDE, INPUT_H // HEATMAP_STRIDE
SIGMA = 2.0  # heatmap 高斯半徑 (heatmap 像素)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def list_samples(split: str) -> list[dict]:
    """回傳該 split 的樣本清單 (正樣本含標註,負樣本 corners=None)。
    切分成員由 data/splits/<split>.txt 明確指定 (見 scripts/make_splits.py)。"""
    stems = (SPLITS_DIR / f"{split}.txt").read_text().split()
    samples = []
    for stem in stems:
        frame = FRAMES_DIR / f"{stem}.jpg"
        if not frame.exists():
            continue
        ann_path = ANNOTATIONS_DIR / f"{stem}.json"
        ann = json.loads(ann_path.read_text()) if ann_path.exists() else None
        samples.append({"path": frame, "ann": ann})
    return samples


def random_homography(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """縮放 + 平移 + 小透視擾動的合成 homography (原圖 -> 原圖座標系)。"""
    scale = rng.uniform(0.7, 1.6)
    cx, cy = w / 2, h / 2
    tx = rng.uniform(-0.15, 0.15) * w
    ty = rng.uniform(-0.15, 0.15) * h
    S = np.array([[scale, 0, cx - scale * cx + tx],
                  [0, scale, cy - scale * cy + ty],
                  [0, 0, 1]], np.float64)
    # 四角小幅擾動的透視變換
    jitter = 0.04
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-jitter, jitter, (4, 2)) * [w, h]).astype(np.float32)
    P = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return P @ S


def color_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.integers(-12, 13)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def render_heatmap(cx: float, cy: float) -> np.ndarray:
    """在 heatmap 座標 (cx, cy) 畫高斯峰。
    最近的整數像素強制設為 1.0 (focal loss 以 >0.999 認定正位置)。"""
    xs = np.arange(HM_W, dtype=np.float32)
    ys = np.arange(HM_H, dtype=np.float32)[:, None]
    hm = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * SIGMA ** 2))
    ix = int(np.clip(round(cx), 0, HM_W - 1))
    iy = int(np.clip(round(cy), 0, HM_H - 1))
    hm[iy, ix] = 1.0
    return hm


class TableDataset(Dataset):
    def __init__(self, split: str, augment: bool):
        self.samples = list_samples(split)
        self.augment = augment
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = cv2.imread(str(s["path"]))
        h, w = image.shape[:2]
        ann = s["ann"]

        if ann is not None:
            corners = np.array([ann["corners"][n] for n in CORNER_NAMES], np.float64)
            vis_gt = np.array([ann["visibility"][n] for n in CORNER_NAMES], bool)
        else:
            corners = np.zeros((4, 2))
            vis_gt = np.zeros(4, bool)

        if self.augment:
            H = random_homography(w, h, self.rng)
            image = cv2.warpPerspective(image, H, (w, h), borderValue=(30, 30, 30))
            if ann is not None:
                pts = cv2.perspectiveTransform(corners.reshape(1, 4, 2), H)
                corners = pts.reshape(4, 2)
            if self.rng.random() < 0.5:  # 水平翻轉
                image = image[:, ::-1].copy()
                corners[:, 0] = w - corners[:, 0]
                corners = corners[FLIP_PERM]
                vis_gt = vis_gt[FLIP_PERM]
            image = color_jitter(image, self.rng)

        # 縮到輸入解析度
        sx, sy = INPUT_W / w, INPUT_H / h
        image = cv2.resize(image, (INPUT_W, INPUT_H))
        corners = corners * [sx, sy]

        heatmaps = np.zeros((4, HM_H, HM_W), np.float32)
        vis_target = np.zeros(4, np.float32)
        if ann is not None:
            for i in range(4):
                x, y = corners[i]
                in_frame = 0 <= x < INPUT_W and 0 <= y < INPUT_H
                if in_frame and vis_gt[i]:
                    heatmaps[i] = render_heatmap(x / HEATMAP_STRIDE - 0.5,
                                                 y / HEATMAP_STRIDE - 0.5)
                    vis_target[i] = 1.0

        tensor = image[..., ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1))

        return {
            "image": tensor,
            "heatmaps": torch.from_numpy(heatmaps),
            "presence": torch.tensor(1.0 if ann is not None else 0.0),
            "visibility": torch.from_numpy(vis_target),
            "corners": torch.from_numpy(corners.astype(np.float32)),  # 輸入座標系
            "vis_gt": torch.from_numpy(vis_gt.astype(np.float32)),    # 原標註可見性
            "stem": s["path"].stem,
        }
