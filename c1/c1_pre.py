'''
使用過濾後的advanced_filter.csv做前處理，矩陣轉換之類的東西，生成訓練集與驗證集。
最終檔案產出為C1_ML_Training_X.npz, C1_ML_Training_y.npy 等
'''

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.utils import shuffle
import joblib
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 控制面板：模型特徵選擇
# ---------------------------------------------------------
# 備註：在下面設定是否將該系列特徵放入訓練矩陣 X 中
CONFIG_FEATURES = {
    'Expected_LeadTime': True,        # 預計進貨天數 (Expected_LeadTime_Log)
    'LeadTime_Include_Holiday': False, # 進貨天數(包含非工作日) (LeadTime_Holiday_PT)
    'LeadTime_Hist': True,            # 歷史進貨天數統計 (Hist_Mean_LeadTime 等)
    'LeadTime_Actual_Hist': True,     # 歷史實際進貨天數統計 (Hist_Actual_Mean, Hist_Actual_CV 等)
    'LeadTime_MA': True,              # 最近幾次的進貨天數MA (LeadTime_MA3, MA5)

    'Hash_ID_Full': True,             # 14碼完整品號雜湊
    'Hash_ID_Split': True,            # 分段品號雜湊
    'Hash_Spec': True,                 # 品名規格雜湊

    'Amount': True,                   # 採購數量 (Amount_PT)

    'Category_OneHot': True          # 品號前綴分類 (Category_M0 等)
}


# ==============================================================================
# 1. 讀取資料
# ==============================================================================
try:
    # 優先讀取 c1_filter.py 產出的 CSV
    df_raw = pd.read_csv('C1_advance_filter.csv')
    df_stats = pd.read_csv('C1_Stats_Filtered.csv')
except Exception:
    df_raw = pd.read_excel('C1_advance_filter.xlsx')
    df_stats = pd.read_excel('C1_Stats_Filtered.xlsx')

df_raw.columns = [str(c).strip() for c in df_raw.columns]
df_stats.columns = [str(c).strip() for c in df_stats.columns]

# ==============================================================================
# 2. 合併歷史統計特徵 (History Stats)
# ==============================================================================
cols_to_merge = ['品名', 'mean', 'std', 'count', 'min', 'max', 'amount', 'actual_mean', 'actual_std', 'actual_min', 'actual_max', 'actual_cv']
# 容錯處理，若找不到 amount 則補 0
if 'amount' not in df_stats.columns:
    df_stats['amount'] = 0
cols_to_merge = [c for c in cols_to_merge if c in df_stats.columns]

df = pd.merge(df_raw, df_stats[cols_to_merge], on='品名', how='left')

df = df.rename(columns={
    'mean': 'Hist_Mean_LeadTime', 'std': 'Hist_Std_LeadTime', 
    'count': 'Hist_Purchase_Count', 'min': 'Hist_Min_LeadTime', 'max': 'Hist_Max_LeadTime',
    'amount': 'Hist_Purchase_Amount',
    'actual_mean': 'Hist_Actual_Mean', 'actual_std': 'Hist_Actual_Std',
    'actual_min': 'Hist_Actual_Min', 'actual_max': 'Hist_Actual_Max',
    'actual_cv': 'Hist_Actual_CV'
}).fillna(0)

hist_cols = ['Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count', 'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount']

# ==============================================================================
# 3. 計算 MA 移動平均 (依採購日期排序)
# ==============================================================================
df['採購日期'] = pd.to_datetime(df['採購日期'])
df = df.sort_values(by=['品號', '採購日期'])

col_leadtime = next((c for c in df.columns if c in ['進貨天數(已扣除假日)', '實際進貨天數', '進貨天數']), '進貨天數(已扣除假日)')   # 預測目標
df['LeadTime_MA3'] = df.groupby('品號')[col_leadtime].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1)).fillna(0)
df['LeadTime_MA5'] = df.groupby('品號')[col_leadtime].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1)).fillna(0)

# ==============================================================================
# 4. 資料切分 進行 Data Splitting，分成20% validation 與 80% testing。K-fold 沒有更深入做 PT λ 的重新計算。
# ==============================================================================
print("正在洗牌與切分資料 indices...")
indices = np.arange(df.shape[0])
np.random.seed(42)
np.random.shuffle(indices)

idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
df_train = df.iloc[idx_train]

# ==============================================================================
# 5. 特徵工程 (分離數值型與稀疏矩陣型)
# ==============================================================================
print("開始執行特徵抽離與稀疏矩陣轉換 (Sparse Matrix)...")
num_features = []

# (A) 數值型特徵獨立處理 
col_amount = next((c for c in df.columns if '採購數量' in c), None)
if col_amount:
    pt_amount = PowerTransformer(method='yeo-johnson', standardize=True)
    # 僅對 df_train fit，避免 Data Leak
    pt_amount.fit(df_train[col_amount].fillna(0).values.reshape(-1, 1))
    # 對全量資料進行 transform
    df['Amount_PT'] = pt_amount.transform(df[col_amount].fillna(0).values.reshape(-1, 1)).flatten()
    
    lambda_value = pt_amount.lambdas_[0]
    print(f"計算出的 Lambda 值為: {lambda_value}")
    
    joblib.dump(pt_amount, 'amount_power_transformer.joblib')
else:
    df['Amount_PT'] = 0

# 加入預計進貨天數特徵 (Expected Lead Time)
col_elt = next((c for c in df.columns if '預計進貨天數' in c), None)
if col_elt:
    # 預計進貨天數做 log 轉換
    df['Expected_LeadTime_Log'] = np.log1p(pd.to_numeric(df[col_elt], errors='coerce').fillna(0).clip(lower=0))
else:
    df['Expected_LeadTime_Log'] = 0

if '進貨天數(包含非工作日)' in df.columns:
    pt_holiday = PowerTransformer(method='yeo-johnson', standardize=True)
    pt_holiday.fit(df_train['進貨天數(包含非工作日)'].fillna(0).values.reshape(-1, 1))
    df['LeadTime_Holiday_PT'] = pt_holiday.transform(df['進貨天數(包含非工作日)'].fillna(0).values.reshape(-1, 1)).flatten()
else:
    df['LeadTime_Holiday_PT'] = 0

# 將使用者設定放入數值特徵庫
if CONFIG_FEATURES.get('LeadTime_MA', True): num_features.extend(['LeadTime_MA3', 'LeadTime_MA5'])
if CONFIG_FEATURES.get('Amount', True): num_features.append('Amount_PT')
if CONFIG_FEATURES.get('Expected_LeadTime', True): num_features.append('Expected_LeadTime_Log')
if CONFIG_FEATURES.get('LeadTime_Hist', True): num_features.extend(hist_cols)
if CONFIG_FEATURES.get('LeadTime_Actual_Hist', True): 
    actual_hist_cols = ['Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min', 'Hist_Actual_Max', 'Hist_Actual_CV']
    num_features.extend([c for c in actual_hist_cols if c in df.columns])
if CONFIG_FEATURES.get('LeadTime_Include_Holiday', True): num_features.append('LeadTime_Holiday_PT')

# 目標值 y (獨立處理)
pt_y = PowerTransformer(method='yeo-johnson', standardize=True)
# 僅對 y_train fit
pt_y.fit(df_train[col_leadtime].values.reshape(-1, 1))
# 對全量資料 transform
y_all = pt_y.transform(df[col_leadtime].values.reshape(-1, 1)).flatten()
joblib.dump(pt_y, 'target_power_transformer.joblib')

y_train, y_test = y_all[idx_train], y_all[idx_test]

# (B) 類別與雜湊特徵 (保留為稀疏矩陣 Sparse Blocks 以免記憶體爆炸)
sparse_blocks = []

if CONFIG_FEATURES.get('Category_OneHot', True):
    id_str_upper = df['品號'].astype(str).str.upper()
    cat_mat = np.column_stack([
        id_str_upper.str.startswith('M0').astype(np.int8),
        id_str_upper.str.startswith('M2').astype(np.int8),
        id_str_upper.str.startswith('K').astype(np.int8),
        id_str_upper.str.startswith('E').astype(np.int8)
    ])
    sparse_blocks.append(('Category_OneHot', sp.csr_matrix(cat_mat)))

if CONFIG_FEATURES.get('Hash_ID_Full', True):
    hasher_id_full = FeatureHasher(n_features=256, input_type='string')
    hash_full_sp = hasher_id_full.transform(df['品號'].astype(str).apply(lambda x: [x]))
    sparse_blocks.append(('Hash_ID_Full', hash_full_sp))

if CONFIG_FEATURES.get('Hash_ID_Split', True):
    hasher_id = FeatureHasher(n_features=128, input_type='string')  # 2 4 4 4
    id_str_pad = df['品號'].astype(str).apply(lambda x: x.ljust(14, ' '))
    hash_split_sp = sp.hstack([
        hasher_id.transform(id_str_pad.str[0:2].apply(lambda x: [x])),
        hasher_id.transform(id_str_pad.str[2:6].apply(lambda x: [x])),
        hasher_id.transform(id_str_pad.str[6:10].apply(lambda x: [x])),
        hasher_id.transform(id_str_pad.str[10:14].apply(lambda x: [x]))
    ], format='csr')
    sparse_blocks.append(('Hash_ID_Split', hash_split_sp))

if CONFIG_FEATURES.get('Hash_Spec', True):
    df['Name_Spec'] = df['品名'].astype(str) + "_" + df['規格'].astype(str)
    # 維持原本 1024 維，因為稀疏矩陣已不怕記憶體爆炸
    hasher_spec = FeatureHasher(n_features=512, input_type='string')   # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    hash_spec_sp = hasher_spec.transform(df['Name_Spec'].apply(lambda x: [x]))
    sparse_blocks.append(('Hash_Spec', hash_spec_sp))


# ==============================================================================
# 6. 特徵大整合堆疊 (Sparse HStack)
# ==============================================================================
print("針對數值特徵進行標準化與稀疏矩陣合併...")
train_blocks = []
test_blocks = []
full_blocks = []

if num_features:
    X_num = df[num_features].fillna(0).values
    X_num_train = X_num[idx_train]
    X_num_test = X_num[idx_test]
    
    # 數值特徵這時才執行 StandardScaler 以保持統計正確性，且不會破壞稀疏特徵的 0
    scaler_x = StandardScaler()
    X_num_train_scaled = scaler_x.fit_transform(X_num_train)
    X_num_test_scaled = scaler_x.transform(X_num_test)
    joblib.dump(scaler_x, 'x_scaler_num.joblib')
    
    train_blocks.append(sp.csr_matrix(X_num_train_scaled))
    test_blocks.append(sp.csr_matrix(X_num_test_scaled))
    
    # 新增：完整特徵集縮放 (用於 K-Fold)
    scaler_x_full = StandardScaler()
    X_num_full_scaled = scaler_x_full.fit_transform(X_num)
    full_blocks.append(sp.csr_matrix(X_num_full_scaled))
    
    print(f"   數值特徵 ({len(num_features)} 維) 處理並標準化完成")

for name, block in sparse_blocks:
    train_blocks.append(block[idx_train])
    test_blocks.append(block[idx_test])
    full_blocks.append(block)
    print(f"   加入稀疏特徵區塊: {name} (維度: {block.shape[1]})")

# 最終特徵合併
X_train_final = sp.hstack(train_blocks, format='csr')
X_test_final = sp.hstack(test_blocks, format='csr')
X_full_final = sp.hstack(full_blocks, format='csr')
print(f"特徵矩陣建置完成！最終 Train 大小: {X_train_final.shape}")

# ==============================================================================
# 7. 匯出神經網路訓練檔案 (.npz & .npy)
# ==============================================================================
print("正在匯出 ML 訓練檔案 (NPZ 格式解省記憶體與硬碟)...")
sp.save_npz('C1_ML_Training_X.npz', X_train_final)
np.save('C1_ML_Training_y.npy', y_train)

sp.save_npz('C1_ML_Test_X.npz', X_test_final)
np.save('C1_ML_Test_y.npy', y_test)

# 匯出 val set 的承諾交期天數，供 c1_models.py 計算 Late Recall
col_required = '預計進貨天數'
if col_required in df.columns:
    required_days_test = df[col_required].fillna(0).values[idx_test]
    np.save('C1_ML_Test_required_days.npy', required_days_test)
    print(f"已匯出 val set 承諾交期天數: C1_ML_Test_required_days.npy ({len(required_days_test)} 筆)")
else:
    print(f"警告：找不到欄位 '{col_required}'，Late Recall 計算將無法使用")

# 新增：匯出完整特徵集，提供後續 K-Fold Cross Validation 獨立使用
print("正在匯出 K-Fold 全量資料檔案...")
sp.save_npz('C1_ML_Full_X.npz', X_full_final)
np.save('C1_ML_Full_y.npy', y_all)

# 產出 Debug 用 CSV (只取前 100 筆避免當機，並還原部分資訊)
debug_df = df.iloc[idx_train[:100]].copy()
debug_df['Target_PT'] = y_train[:100]
debug_df.to_csv('C1_ML_Training_Debug_Top100.csv', index=False, encoding='utf-8-sig')

print("所有前處理與矩陣儲存完成！產出檔案：C1_ML_Training_X.npz, C1_ML_Training_y.npy 等。")
