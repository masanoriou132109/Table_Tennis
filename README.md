# 桌球轉播畫面 — 球桌桌面範圍偵測

輸入桌球比賽轉播影片,即時輸出球桌桌面的 4 個角點座標(可再算 homography 做視角矯正)。
轉播畫面會有 zoom in/out 與角度移動,因此採「4 角點 + 可見性」的標註與模型設計。

## 角點順序約定(重要)

所有標註與模型輸出一律遵守固定語意順序,這是計算 homography 的依據:

| 索引 | 名稱 | 說明 |
|------|------|------|
| 1 | `far_left` | 遠側左角(畫面上方那側) |
| 2 | `far_right` | 遠側右角 |
| 3 | `near_right` | 近側右角 |
| 4 | `near_left` | 近側左角 |

角點座標允許超出畫面範圍(zoom in 時常見);被遮擋或出畫面的角以 `visibility: false` 標記。

### Homography 物理對應(依拍攝軸向)

角點順序定義在影像空間,不受鏡頭軸向影響;但算 homography 時,影像角點對應到球桌物理矩形
(2.74 m × 1.525 m)的方式**取決於拍攝軸向**:

- **短軸側拍**(現代 WTT 主鏡頭,裁判在畫面中央、球員分列左右):`far_left → far_right` 是球桌**長邊**,
  物理對應為 `far_left=(0,0)`、`far_right=(2.74,0)`、`near_right=(2.74,1.525)`、`near_left=(0,1.525)`(單位 m)。
- **長軸端拍**(舊式轉播,球員一近一遠):`far_left → far_right` 是球桌**短邊**,物理矩形需旋轉 90°。

目前 `videos/` 中的素材全為短軸側拍,一律採用短軸對應。若日後加入端拍或 replay 視角素材,
標註需另加 `view_axis` 欄位區分,否則 homography 會旋轉 90° 且比例錯誤。

## 目錄結構

```
videos/               原始轉播影片 (mp4)
data/frames/          抽出的影格 (jpg)
data/annotations/     每格一個 JSON: 4 角座標 + 可見性
scripts/
  extract_frames.py   抽格 (間隔取樣 + 去重複)
  annotate_sam.py     SAM2 輔助標註工具
  convert_labelme.py  labelme 標註 → 本專案角點 JSON
  build_labeling_pool.py  新影片抽格 + 模型預標 (baseline 優先, v1 fallback) → 供 labelme 修正
  infer_video.py      影片推論 (--mode sample 抽樣疊圖 / --mode video 輸出疊框影片)
  eval_model.py       評估模型 (協定同 eval_baseline, 可直接對比)
  train.py            訓練 keypoint 模型
src/                  baseline / 模型 / 時序平滑 (後續階段)
```

## 快速開始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. 抽格
python scripts/extract_frames.py

# 2. 標註 (左鍵點桌面 → SAM 分割 → 自動擬合四邊形 → 拖曳微調 → s 儲存)
python scripts/annotate_sam.py
```

首次執行標註工具會自動下載 SAM2 權重 (`sam2.1_b.pt`,約 300MB);想更快可加 `--model sam2.1_t.pt`。

### 替代方案:用 labelme 標註

```bash
labelme data/frames --output data/labelme \
  --flags far_left,far_right,near_right,near_left --nodata --autosave
```

- 每張圖畫一個 label 為 `table` 的 polygon,恰好 4 點(順序不限,轉換時自動重排)
- 被遮擋的角點在估計位置,並勾選對應 flag(勾選 = visibility false)
- 角出畫面時點在畫面邊緣並勾 flag(labelme 的點不能超出畫面,此為與 SAM 工具的差異)
- 標完執行 `python scripts/convert_labelme.py` 轉成 `data/annotations/` 格式

## 路線圖

- [x] Phase 0-2: 骨架、抽格、標註 (labelme, 59 張含可見性)
- [x] Phase 3: 傳統 CV baseline (`src/baseline_cv.py`, magenta 邊線 HSV 遮罩 + 連通元件 + 四邊形擬合)
- [x] Phase 4 v1: keypoint 模型 (`src/model.py` ResNet18 + deconv heatmap, focal loss, subpixel 解碼)
- [x] Phase 4 v2 資料擴充管線: 5 段完整比賽 (88 分, 多場地) + 模型預標池 (`build_labeling_pool.py`)
- [x] Phase 4 v2 重訓: 216 張標註 / 6 場地,val median 2.0px、<10px 100%,災難性錯誤消失
- [x] Phase 5 時序平滑 (`src/smoothing.py`): 信心門控 EMA + 遮擋角用可見角仿射補算。
      標準遮擋大幅改善;極端 zoom-in (多角出框) 仍難。`infer_video.py --mode video --smooth`
- [x] Phase 5 幾何合理性閘門 (`src/geometry.py`): 拒絕自交叉/比例異常的四邊形。
      主鏡位 216/216 零誤拒;側拍/斜角重播被正確擋掉 (Incheon presence>0.5 中拒絕 23%)。
      定位: 回合都在主鏡位,側拍僅重播 → 閘門擋垃圾框即可,不另訓練側拍。
- [x] 遮擋補角升級 (`src/smoothing.py`): 遮擋角改用「可見角平面變換 × 相機運動量」的 hybrid
      補算 (3角→仿射/2角→相似,依運動量在沿用↔變換間混合)。取代原本全程最差的純相似。
- [x] 影片級追蹤 `src/tracker.py` (TableTracker): 整合平滑 + 三重穩健性閘門
      (幾何 / 信心 / 時序確認),廣角·斜角·分布外鏡頭寧可不畫也不畫錯。
- [x] Homography 地基 `src/homography.py`: 每幀 4 角 + 已知桌面尺寸 (2.74×1.525m)
      → 影像像素 ↔ 真實桌面座標。`infer_video.py --grid` 疊真實座標網格驗證。
      落點分析地基: 反彈幀球影像座標 → `image_to_table` → 真實桌面落點。
- [x] 持久先驗補角: 主相機機位固定 → 桌角整場位置穩定 (std~10px)。每次 4 角全可信即記下
      完整四邊形當先驗;被遮時用「先驗可見角→本格可見角」仿射對齊補出被遮角 (透視正確,
      合成測試 ~5px,遠優於單幀平行四邊形 ~49px)。只要整場看過桌子一次,之後每個回合
      開頭發球遮角也能立刻畫框 (滿足「一開始就有框」)。`src/smoothing.py` reference 機制。
- [x] Core ML 匯出 (`scripts/export_coreml.py`): 正規化與 sigmoid 內建於模型,
      Swift 端餵 CVPixelBuffer 直接拿機率。ANE 1.6ms/幀,與 PyTorch 角點差異 0.026px。
- [ ] 後續: TableTracker 移植 Swift → 接上 PingPongTracker app 的桌角校正 (取代手動點角)

## Core ML 部署 (`Models/TableDetector.mlpackage`)

```bash
python scripts/export_coreml.py            # 產生 Models/TableDetector.mlpackage
```

| 項目 | 值 |
|------|-----|
| 模型大小 | 29 MB |
| 推論 (Neural Engine) | **1.6 ms/幀 (638 fps)** |
| 推論 (CPU only) | 10.6 ms/幀 (94 fps) |
| 與 PyTorch 角點差異 | 0.026 px (12 張真實影格,可忽略) |

介面:
- 輸入 `image`: 512×288 RGB (ImageType,0~255);正規化已內建
- 輸出 (皆已過 sigmoid): `heatmaps` [1,4,72,128]、`presence` [1]、`visibility` [1,4]

Swift 端需自行實作: heatmap argmax → 5×5 鄰域加權質心 (subpixel) → ×stride 4 → 乘回原圖尺度,
再套 `src/tracker.py` 的平滑/先驗補角/三重閘門 (設計上純幾何,無顏色依賴,可直接移植)。

搭配 PingPongTracker app 時注意角點順序: 本專案為 `far_left, far_right, near_right, near_left`,
該 app 為 `near-left, near-right, far-right, far-left` — 兩者互為反序 (`corners.reversed()`)。

## 影片級穩健性 (TableTracker)

單幀模型三類錯誤,各一道閘門 (`src/tracker.py`):

| 錯誤類型 | 閘門 | 訊號 |
|---------|------|------|
| 自交叉/比例異常四邊形 | 幾何 (`is_plausible_quad`) | 凸性/長寬比/面積 |
| 廣角·斜角·分布外 (整體不確定) | 信心 | 可信角數 ≥2 (廣角段角信心 p10=0.12 vs 主鏡位 0.86) |
| 瞬態跳變/場景切換殘影 | 時序確認 | 連續 3 幀一致才輸出,大跳變重啟追蹤 |

繪製率驗證: Doha 主鏡位段 64%、Chongqing 廣角段 14% (廣角垃圾擋掉 86%,主鏡位保留)。
純幾何+狀態機,不依賴顏色,設計上可直接移植到 Swift 供 Core ML 部署。

## 遮擋補角方法評估 (`scripts/eval_occlusion_fill.py`, 合成持續遮擋, 誤差 px)

單一剛性矩形被擋一角時,3 影像角+已知矩形無法唯一反推第 4 角 (homography 8 DOF 需 4 點),
故採時序幾何錨定。純幾何、場地無關、可隨模型部署 (不依賴顏色/邊線偵測)。

| 遮擋幀數 | hold | similarity | affine | **hybrid** |
|---------|------|-----------|--------|-----------|
| Chongqing (運鏡) 10 | 3.3 | 8.4 | 5.0 | **4.0** |
| Chongqing (運鏡) 30 | 6.3 | 13.7 | 5.5 | **5.9** |
| Doha (靜止) 30 | 2.4 | 8.3 | 4.0 | **3.4** |

結論: 純相似 (原方案) 全程最差 (槓桿放大抖動);hold 靜止時好但運鏡長遮擋會漂;
hybrid 依相機運動量混合,兩情境皆穩健,已採用。
- [ ] Phase 5: 時序平滑 (Kalman / optical flow) 與評估 (角點像素誤差、重投影誤差)
- [ ] 部署: coremltools → Core ML → iOS/macOS 即時推論

## Baseline 成績 (2026-08-30, 1280x720, 可見角點像素誤差)

切分: 按場次,val = 澳門站 20230418 (跨場地泛化)。評估: `python scripts/eval_baseline.py`

| Split | 偵測率 | mean | median | <5px | <10px |
|-------|--------|------|--------|------|-------|
| train (法蘭克福 x3 場) | 40/41 | 12.0 | 2.0 | 85% | 90% |
| val (澳門) | 16/18 | 17.2 | 5.6 | 32% | 89% |

已知限制 (Phase 4 模型的改進目標):
- 澳門場邊線較粗/光暈強,偵測框系統性外偏約 3-5px (val <5px 偏低的主因)
- 球員大面積遮擋邊線時偵測失敗或角點大偏 (3 張失敗、p95 誤差大的來源)
- 桌面貼近畫面邊界 (zoom in) 時會被邊界過濾規則誤殺

## Keypoint 模型 v1 成績 (訓練: `scripts/train.py`, 評估: `scripts/eval_model.py`)

400 epochs, focal loss, MPS 約 11 分鐘。checkpoint: `checkpoints/best.pt` (epoch 365)。

| Split | presence | median | <10px | p95 |
|-------|----------|--------|-------|-----|
| train | 100% | 5.0 | 72% | 412 |
| val (澳門) | 100% | 4.5 | 80% | 380 |

對比 baseline: median 已勝出 (4.5 vs 5.6),且能判斷「畫面無桌面」(baseline 不行)。

v1 已知問題 (v2 目標):
- 約 1/4 的角點有 >30px 災難性錯誤,主因是右側/近側角通道混淆 (heatmap 峰跑到別的角)
- heatmap 峰值分數與 visibility head 都無法區分好壞預測 (信心訊號不可用),
  錯誤無法靠閾值過濾 —— 41 張訓練圖不足以學會角點消歧,需 pseudo-labeling 擴充資料
- MSE loss 版本 (v0) val median 15.9px,focal loss + subpixel 解碼帶來 3.5 倍改善

## Keypoint 模型 v2 成績 (216 張 / 6 場地, checkpoint epoch 365)

| Split | presence | median | mean | <5px | <10px | p95 | max |
|-------|----------|--------|------|------|-------|-----|-----|
| train | 100% | 2.0 | 2.2 | 98% | 100% | 4.3 | 11.1 |
| val | 100% | 2.0 | 2.4 | 95% | 100% | 5.0 | 28.8 |

關鍵改善: v1 的災難性錯誤 (val p95 380px / max 398px) 幾乎消失 (p95 5.0 / max 28.8)。
多場地資料 (藍桌/粉紅桌/magenta 桌) 解決了 v1 的角點消歧問題。v1 權重備份於 `checkpoints/v1_best.pt`。
