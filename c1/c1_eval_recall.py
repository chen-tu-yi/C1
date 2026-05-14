"""
c1_eval_recall.py
從已儲存的模型與 val set 檔案直接計算三種 Recall，不需要重新訓練。

需要的檔案（由 c1_pre.py 與 c1_models.py 產出）：
  - XGBoost_Model.joblib
  - target_power_transformer.joblib
  - C1_ML_Test_X.npz
  - C1_ML_Test_y.npy
  - C1_ML_Test_required_days.npy

  /c1
      ├── XGBoost_Model.joblib  
      ├── target_power_transformer.joblib  
      ├── C1_ML_Test_X.npz  
      ├── C1_ML_Test_y.npy  
      ├── C1_ML_Test_required_days.npy  
      ├── C1_Model_Features.joblib  # <--- 這裡
      └── C1_Model_Features_Lookup.csv
"""

import numpy as np
import scipy.sparse as sp
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# =========================================================
# 1. 載入模型與資料
# =========================================================
print("載入模型與 val set...")
model = joblib.load('XGBoost_Model.joblib')
pt_y = joblib.load('target_power_transformer.joblib')

X_val = sp.load_npz('C1_ML_Test_X.npz')
y_val = np.load('C1_ML_Test_y.npy')
required_days_val = np.load('C1_ML_Test_required_days.npy')

print("重建資料切分以對齊品號...")
df_raw = pd.read_csv('C1_advance_filter.csv')
df_raw.columns = [str(c).strip() for c in df_raw.columns]
if '採購日期' in df_raw.columns:
    df_raw['採購日期'] = pd.to_datetime(df_raw['採購日期'])
elif '單據日期' in df_raw.columns:
    df_raw['單據日期'] = pd.to_datetime(df_raw['單據日期'])
    
date_col = '採購日期' if '採購日期' in df_raw.columns else '單據日期'
df = df_raw.sort_values(by=['品號', date_col])
indices = np.arange(df.shape[0])
idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
test_items = df['品號'].iloc[idx_test].astype(str).str.strip().values

print("讀取 INVMB 過濾屬性為 'P' 的物料...")
erp_path = r'C:\local_file\專題\ERP_Table.xlsx'
if os.path.exists(erp_path):
    try:
        invmb_df = pd.read_excel(erp_path, sheet_name='INVMB', usecols=['品號 (MB)', '品號屬性 (MB)'])
        valid_items = invmb_df[invmb_df['品號屬性 (MB)'].astype(str).str.strip().str.upper() == 'P']['品號 (MB)'].astype(str).str.strip().tolist()
        mask = np.isin(test_items, valid_items)
    except Exception as e:
        print(f"讀取 INVMB 發生錯誤: {e}，將評估所有測試集。")
        mask = np.ones(len(test_items), dtype=bool)
else:
    print(f"找不到 {erp_path}，將評估所有測試集。")
    mask = np.ones(len(test_items), dtype=bool)

print(f"過濾前測試集筆數: {len(test_items)}, 過濾後筆數: {mask.sum()}")

# =========================================================
# 2. 預測並還原至天數空間
# =========================================================
print("進行預測...")
y_pred = model.predict(X_val)
y_val_inv = pt_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_pred_inv = pt_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# =========================================================
# 3. 計算三種 Recall
# =========================================================
# 只保留屬性為 'P' 的預測結果
y_val_inv = y_val_inv[mask]
y_pred_inv = y_pred_inv[mask]
req = required_days_val[mask].astype(float)

actual_late  = y_val_inv > req + 1
actual_early = y_val_inv < req
actual_nonot = actual_late | actual_early
actual_ontime = ~actual_nonot

predicted_late   = y_pred_inv > req + 1
predicted_early  = y_pred_inv < req
predicted_nonot  = predicted_late | predicted_early
predicted_ontime = ~predicted_nonot

# Late Recall（只有遲到）
n_late = actual_late.sum()
tp_late = (actual_late & predicted_late).sum()
late_recall = tp_late / n_late if n_late > 0 else float('nan')

# Non-ontime Recall（遲到 + 提早）
n_nonot = actual_nonot.sum()
tp_nonot = (actual_nonot & predicted_nonot).sum()
nonot_recall = tp_nonot / n_nonot if n_nonot > 0 else float('nan')

# Ontime Recall（準時）
n_ontime = actual_ontime.sum()
tn = (actual_ontime & predicted_ontime).sum()
ontime_recall = tn / n_ontime if n_ontime > 0 else float('nan')

# =========================================================
# 4. 輸出結果
# =========================================================
print("\n====== Recall 計算結果 ======")
print(f"Late Recall     (只算遲到):  {late_recall:.4f}  (TP={tp_late}, TP+FN={n_late})")
print(f"Non-ontime Recall (遲到+提早): {nonot_recall:.4f}  (TP={tp_nonot}, TP+FN={n_nonot})")
print(f"Ontime Recall   (準時):       {ontime_recall:.4f}  (TN={tn}, TN+FP={n_ontime})")
print("=============================")
