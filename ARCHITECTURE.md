# 系統架構

桌球轉播影片的**球桌桌面偵測**,以及與教授 `PingPongTracker` 專案串接的**落點分析**。

---

## 1. 全貌:兩個 repo 的分工

```
┌──────────────────────────────────────┐     ┌────────────────────────────────────┐
│  本專案 (Table_Tennis)                │     │  教授 (PingPongTracker)             │
│  ─────────────────────                │     │  ────────────────────               │
│  桌面偵測 · 逐幀 4 角點                │     │  球偵測 (YOLO26 Core ML)            │
│  時序追蹤 · 遮擋補角                   │     │  擊球/落點事件演算法                 │
│  每幀 homography                      │     │  iOS App (SwiftUI)                 │
└──────────────┬───────────────────────┘     └────────────────┬───────────────────┘
               │          scripts/landing_points.py           │
               └──────────────────►  串接  ◄───────────────────┘
                              (不修改對方程式碼)
```

**分工要點**:他的 iOS app 原本要**使用者手動點 4 個桌角**校正,離線工具 `auto_corners.py`
則假設固定機位(整支影片一組角點)。本專案提供**逐幀自動桌角**,因此轉播鏡頭
zoom/pan 時映射仍正確 — 這是兩邊結合的核心價值。

---

## 2. 資料流

```
轉播影片 (1280x720)
    │
    ├─► TableKeypointNet (Core ML / PyTorch)
    │       └─► heatmaps[4] + presence + visibility
    │              │
    │              ├─► decode: argmax → 5x5 加權質心 (subpixel) → ×stride4
    │              │
    │              └─► TableTracker ── 三重閘門 ─────────────┐
    │                     ├ 幾何 (凸性/長寬比/面積)           │
    │                     ├ 信心 (≥2 角 score≥0.75)          │
    │                     ├ 時序 (連續 3 幀一致)              │
    │                     └ 持久先驗補角 (遮擋時)             │
    │                                                       ▼
    │                                          逐幀 4 角 → homography H(t)
    │                                                       │
    └─► BallDetector.mlpackage (教授, YOLO26@640)            │
            └─► 每幀球候選 ──── 場外閘門 (用 H(t)) ◄──────────┤
                                     │                      │
                                     ▼                      │
                    教授的 build_track → detect_events       │
                    (斜率反轉 → 兩段拋物線擬合 → 交點)        │
                                     │                      │
                                     ▼                      │
                              落點 (次幀級影像座標)            │
                                     │                      │
                                     └── map_point(H(t)) ◄───┘
                                                │
                                                ▼
                                  桌面座標 (cm) + 3x3 分區
```

---

## 3. 桌面偵測子系統(本專案核心)

### 3.1 資料

| 項目 | 內容 |
|------|------|
| 影片 | 15 支(9 段短片 + 5 場完整比賽 + 1 webm) |
| 標註 | **216 張 / 6 場地**(Frankfurt, Macau, Chongqing, Doha, Vegas, Incheon) |
| 負樣本 | 27 張無桌面特寫(訓練 presence head) |
| 切分 | 場地感知:每場地內時序隔離,6 場地都出現在 train/val(182 / 61) |

**標註順序約定**:`far_left, far_right, near_right, near_left`(遠 = 畫面上方)。
教授的順序為 `near-left, near-right, far-right, far-left` — **兩者互為反序**。

### 3.2 模型 `src/model.py`

```
輸入 512x288 RGB
  → ResNet18 backbone (stride 32)
  → 3x ConvTranspose (stride 32 → 4)
  → heatmap head:  [4, 72, 128]   4 個角的 heatmap
  → cls head:      presence[1] + visibility[4]   (由 backbone global pool)
```

- **Loss**:CenterNet 式 penalty-reduced focal loss(比 MSE 好 3.5 倍,小資料下差異顯著)
- **解碼**:argmax 找峰 → 5×5 鄰域機率加權質心(subpixel,消除 stride-4 量化誤差)

### 3.3 時序追蹤 `src/tracker.py` + `src/smoothing.py` + `src/geometry.py`

純幾何 + 狀態機,**不依賴顏色**,設計上可直接移植 Swift。

| 機制 | 作用 |
|------|------|
| **持久先驗** | 4 角全可信時記下模板(主相機機位固定,角位置 std~10px);跨回合有效 |
| **遮擋補角** | 先驗 + 可見角仿射對齊,依相機運動量在「先驗原位 ↔ 仿射」混合,再 EMA |
| **幾何閘門** | 拒絕自交叉/比例異常四邊形(由 216 張標註校準) |
| **信心閘門** | 可信角數 ≥2(門檻 0.75) |
| **時序閘門** | 連續 3 幀一致才輸出;大跳變重啟追蹤 |

---

## 4. 模組職責

### `src/`

| 檔案 | 職責 |
|------|------|
| `model.py` | TableKeypointNet、heatmap 解碼、focal loss |
| `dataset.py` | 資料集、增強(zoom/透視/翻轉/色彩抖動)、heatmap 生成 |
| `tracker.py` | **TableTracker** — 整合平滑與三重閘門(部署時的主入口) |
| `smoothing.py` | EMA 平滑、持久先驗、遮擋補角 |
| `geometry.py` | 四邊形合理性判定 |
| `homography.py` | 影像 ↔ 真實桌面座標(2.74×1.525 m)、桌面網格繪製 |
| `baseline_cv.py` | 傳統 CV baseline(magenta 邊線),已被模型取代,保留作對照 |

### `scripts/`

**資料準備**
| 檔案 | 用途 |
|------|------|
| `extract_frames.py` | 影片抽格(間隔取樣 + 去重複) |
| `build_labeling_pool.py` | 新影片 → 模型預標池(baseline 優先,v1 fallback) |
| `relabel_doha_pink.py` | Doha 粉紅桌專用預標(粉紅色塊法) |
| `convert_labelme.py` | labelme → 本專案角點 JSON |
| `make_splits.py` | 場地感知 train/val 切分 |
| `annotate_sam.py` | SAM2 輔助標註(備用路徑) |

**訓練與評估**
| 檔案 | 用途 |
|------|------|
| `train.py` | 訓練(focal loss, 400 epochs, MPS) |
| `eval_model.py` / `eval_baseline.py` | 模型 / baseline 評估(同一協定可對比) |
| `eval_occlusion_fill.py` | 合成遮擋測試,比較補角策略 |
| `visualize_annotations.py` | 標註品質檢查 |

**推論與部署**
| 檔案 | 用途 |
|------|------|
| `infer_video.py` | 影片推論(`--smooth` 追蹤 / `--grid` 座標網格 / `--show-scores`) |
| `export_coreml.py` | → `Models/TableDetector.mlpackage` |

**落點分析(與教授串接)**
| 檔案 | 用途 |
|------|------|
| `landing_points.py` | 端到端:逐幀桌角 + 球偵測 + 事件演算法 → 落點 JSON |
| `plot_landing_map.py` | 落點俯視圖 + 分區統計 |
| `render_landing_demo.py` | 示範影片(桌面框 + 右下落點小視窗) |

---

## 5. 目前效能

### 桌面偵測 (v2, 216 張 / 6 場地)

| Split | presence | median | <10px | p95 |
|-------|----------|--------|-------|-----|
| train | 100% | 2.0 px | 100% | 4.3 |
| val | 100% | **2.0 px** | **100%** | 5.0 |

### Core ML 部署

| 項目 | 值 |
|------|-----|
| 模型 | 29 MB |
| Neural Engine | **1.6 ms/幀 (638 fps)** |
| CPU only | 10.6 ms/幀 |
| 與 PyTorch 差異 | 0.026 px |

### 落點分析 (Doha 40s)

| 項目 | 結果 |
|------|------|
| 有主鏡位桌面的幀 | 64% |
| 落點偵測 | 16 個,**全部成功映射**,桌外誤判 0 |
| 軌跡完整時的落點召回 | 86% |
| 軌跡稀疏時 | 24% ← **瓶頸** |

---

## 6. 已知限制與瓶頸

| 限制 | 說明 | 狀態 |
|------|------|------|
| **球偵測覆蓋率** | 教授的模型在轉播畫面回合中僅 66% 幀抓到球,軌跡稀疏導致落點漏判約 40% | **最大瓶頸** |
| 側拍/廣角鏡頭 | 訓練資料只有主鏡位,分布外 → 閘門擋下不繪製(寧可不畫也不畫錯) | 已界定範疇 |
| 首次建立先驗前 | 整支影片第一次完整看到桌子之前不繪製 | 長影片幾乎無感 |
| 遮擋角精度 | 被遮角無直接影像證據,估計有不確定性下限 | 已最佳化 |

**已驗證無效的嘗試**(避免重複走):
- 降低球偵測門檻 0.3→0.1:只多 8% 偵測
- 裁切桌面區域放大球:反而略差(超出模型訓練尺度)
- 單幀幾何補角(平行四邊形):透視下誤差 ~49px,不可行

---

## 7. 下一步選項

1. **球偵測微調** — 用轉播畫面標註球位置 fine-tune,直攻最大瓶頸
2. **改用運動線索偵測器** — TrackNet 類(吃連續 3 幀),小快模糊球的標準解
3. **Swift 移植** — `TableTracker` → Swift,接上 iOS app 取代手動點角校正

### 待教授修正(他的 repo)

| 檔案 | 行 | 問題 |
|------|-----|------|
| `Tools/detect_events.py` | 33 | `TABLE_W, TABLE_H = 500.0, 240.0` → 應為 274 × 152.5 |
| `Sources/UI/TrackerViewModel.swift` | 13 | `tableSize = CGSize(width: 500, height: 240)` 同上 |

(本專案已於 `landing_points.py` 內以 override 修正,不影響他的檔案；
 Swift 那處連帶影響 L233-234 的場外閘門 `expand`,改尺寸時需一併按比例調整。)
