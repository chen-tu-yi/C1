# Project Roadmap - AI 供應鏈延遲預測與瓶頸分析

本專案旨在透過機器學習模型（C1 系列）預測物料進貨延遲，並結合視覺化瓶頸分析（C2 系列）優化供應鏈效率。

## 🚀 技術棧 (Tech Stack)

### 核心語言與框架

- **語言**: Python 3.x
- **資料處理**: Pandas, NumPy
- **機器學習模型**:
  - XGBoost (當前表現最佳 RMSE: 0.0545)
  - CatBoost
  - LightGBM
  - Neural Networks (Deep Learning)
- **視覺化**: Matplotlib, Seaborn

### 工程工具

- **特徵工程**: One-hot Encoding, Z-score Normalization, Hashing Vectorization (品名/規格處理)
- **統計分析**: K-means Clustring (標註異常值), moving average, standard deviation

---

## 📋 功能規劃與進展

### Phase 1: 資料整合與特徵工程 (已完成)

- [x] 資料清洗：剔除預計進貨天數 > 90 天之單據。
- [x] 特徵轉換：進貨天數對數化處理 (Log transformation)。
- [x] 特徵擴增：新增移動平均延遲、歷史波動性 (Std) 等績效指標。
- [x] 進階 Embedding：嘗試將「品名+規格」透過 LM 轉為向量。

### Phase 2: 模型開發與效能基準 (進行中)

- [x] 建立 Baseline 模型：XGBoost, LightGBM, CatBoost。
- [x] 神經網路模型實作：NN 訓練流程建立。
- [x] 模型指標對齊：導入 scipy.sparse 加速特徵讀取，並實作 Inverse Metrics 取得天數維度的真實誤差。
- [ ] 超參數調優：使用 Optuna 進行超參數調優。

### Phase 3: 視覺化分析與瓶頸識別 (已完成)

- [x] 風險地圖產出：C1_4_RiskMap, C2_Visual_Analysis。
- [x] 瓶頸分析：C2 人力 (Manpower) 與物料 (Material) 延遲分析。
- [x] 分布與對比圖：分布圖、圓餅圖、盒鬚圖 (BoxPlot)。
- [x] 對高風險物料做標記（實作跨天數風險分級，加註視覺化警示符號）。

### Phase 4: 資料庫建立與高級彙整 (已完成)

- [x] **MOCTA/MOCTB 材料需求最大化**:
  - 實作「產品品名」對「材料」的彙整邏輯。
  - 抓出同產品品名在不同製令單下，所有出現在同規格下的物料最大數量（已領用量）。
- [x] **材料/製令關聯建表**:
  - 建立「材料品號」對「品名」以及「製令單號」對「材料」的對應關係。
- [x] **總機計畫清洗與統計**:
  - 針對預計開工、製程名稱進行清洗與合併 (`shortage_material.py` / `ex_to_model.py`)。
- [x] **全視角製令總表整併 (`all_table.py`)**:
  - 將 AI 延遲預測合併回實體物料庫存需求，產出整合觀測表 `製令總表.csv`。
- [x] **資料庫整合與讀寫架構 (`create_db.py`)**:
  - 建立 SQL Server (MSSQL) 連線，把梳理好的製程物料清單、預測風險表單自動化掃入資料庫。

### Phase 5: 系統整合與部署準備 (進行中)

- [x] **推論管線建立**: 模組化執行 `model_predict.py` 產出推論結果。
- [x] **介面開發**: 介面已完全開發完畢。
- [x] **介面配合**: 提供完整處理過的 CSV 分級檔案與資料庫架構供 Frontend / ERP 呼叫。
- [ ] 自動排程執行 (Cron / Task Scheduler) 介接。
- [ ] 持續學習 (Continuous Learning)：建立模型依據新資料自動重新調優與訓練的機制。

### 待辦清單 (尚未執行的核心任務)

- [ ] 模型超參數調優 (Optuna) 以推進 RMSE/MAE 穩定度。
- [ ] 完整建立 MLOps 自動重新訓練流程。
- [ ] 進一步與 ERP API 達到即時雙向數據同步 (目前仍以檔案為媒介)。

---

## 📈 里程碑 (Milestones)

1. **2025 Q1**: 完成基礎資料清洗與 C1/C2 初版模型驗證。
2. **2025 Q2**: 視覺化看板優化，完成風險地圖與瓶頸分析自動化腳本。
3. **2025 Q3**: 模型精準度提升計畫，R2 Score 改善。
4. **2025 Q4**: 完成專案總體報告與技術文件移交。
