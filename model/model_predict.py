'''
model預測缺料物料延誤時間，輸出報告"缺料風險預測報告.csv"與"叫料單.csv"。
'''
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os
import warnings
from datetime import datetime, timedelta
from sklearn.feature_extraction import FeatureHasher

warnings.filterwarnings('ignore')

def get_predictions(df, lookup_df, id_col, name_col, spec_col, amount_col, models):
    # 只取需要的 lookup_df 欄位進行合併，避免欄位名稱衝突 (例如 '品名' 和 '規格')
    lookup_cols = [
        '品號', 'Expected_LeadTime_Median',
        'LeadTime_MA3', 'LeadTime_MA5',
        'Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count', 
        'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount', 
        'Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min', 
        'Hist_Actual_Max', 'Hist_Actual_CV'
    ]
    lookup_subset = lookup_df[lookup_cols]
    
    # 合併查找表 (獲取歷史特徵)
    df_merged = pd.merge(df, lookup_subset, left_on=id_col, right_on='品號', how='left')
    df_merged.fillna(0, inplace=True)

    # ---------------------------------------------------------
    # 2. 數值特徵處理 (必須與 c1_pre.py 的順序完全一致: 10個特徵)
    # ---------------------------------------------------------
    
    # A. 處理數量 (Amount_PT)
    if models['pt_amount']:
        df_merged['Amount_PT'] = models['pt_amount'].transform(df_merged[[amount_col]].fillna(0)).flatten()
    else:
        df_merged['Amount_PT'] = np.log1p(df_merged[amount_col].fillna(0))

    # B. 處理預計進貨天數 (Expected_LeadTime_Log)
    df_merged['Expected_LeadTime_Log'] = np.log1p(df_merged['Expected_LeadTime_Median'].fillna(0).clip(lower=0))

    all_num_cols = [
        'LeadTime_MA3', 'LeadTime_MA5',       
        'Amount_PT',                          
        'Expected_LeadTime_Log',              
        'Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count', 
        'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount', 
        'Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min', 
        'Hist_Actual_Max', 'Hist_Actual_CV'   
    ]
    
    X_num = df_merged[all_num_cols].fillna(0)
    X_num_scaled = models['scaler_x'].transform(X_num)
    
    # ---------------------------------------------------------
    # 3. 類別與雜湊特徵 (Hashing) - 必須與 c1_pre.py 順序一致
    # ---------------------------------------------------------
    sparse_blocks = []
    
    # (A) Category_OneHot (4 dims)
    id_str_upper = df_merged[id_col].astype(str).str.upper()
    cat_mat = np.column_stack([
        id_str_upper.str.startswith('M0').astype(np.int8),
        id_str_upper.str.startswith('M2').astype(np.int8),
        id_str_upper.str.startswith('K').astype(np.int8),
        id_str_upper.str.startswith('E').astype(np.int8)
    ])
    sparse_blocks.append(sp.csr_matrix(cat_mat))
    
    # (B) Hash_ID_Full (256 dims)
    hasher_id_full = FeatureHasher(n_features=256, input_type='string')
    hash_full_sp = hasher_id_full.transform(df_merged[id_col].astype(str).apply(lambda x: [x]))
    sparse_blocks.append(hash_full_sp)
    
    # (C) Hash_ID_Split (128 * 4 = 512 dims)
    hasher_id_split = FeatureHasher(n_features=128, input_type='string')
    id_str_pad = df_merged[id_col].astype(str).apply(lambda x: x.ljust(14, ' '))
    hash_split_sp = sp.hstack([
        hasher_id_split.transform(id_str_pad.str[0:2].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[2:6].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[6:10].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[10:14].apply(lambda x: [x]))
    ], format='csr')
    sparse_blocks.append(hash_split_sp)
    
    # (D) Hash_Spec (512 dims)
    df_merged['Name_Spec'] = df_merged[name_col].astype(str) + "_" + df_merged[spec_col].astype(str).fillna('')
    hasher_spec = FeatureHasher(n_features=512, input_type='string')
    hash_spec_sp = hasher_spec.transform(df_merged['Name_Spec'].apply(lambda x: [x]))
    sparse_blocks.append(hash_spec_sp)

    # 4. 合併為最終稀疏矩陣
    X_final = sp.hstack([sp.csr_matrix(X_num_scaled)] + sparse_blocks, format='csr')

    # 5. 模型預測
    y_pred_pt = models['xgb'].predict(X_final)
    y_pred_days = models['pt_target'].inverse_transform(y_pred_pt.reshape(-1, 1)).flatten()
    
    # 回傳合理天數
    return np.maximum(np.round(y_pred_days), 1).astype(int)


def prepare_and_predict():
    # 1. 設定路徑
    BASE_DIR = r'C:\local_file\專題\c1'
    MODEL_DIR = r'C:\local_file\專題\model'
    
    # 輸入檔案
    INPUT_PATH = os.path.join(MODEL_DIR, 'Model缺料物料.csv')
    LOOKUP_PATH = os.path.join(MODEL_DIR, 'C1_Inference_Features_Lookup.csv')
    PURT_PATH = os.path.join(BASE_DIR, 'PURT.csv')
    
    # 載入訓練好的組件
    SCALER_PATH = os.path.join(BASE_DIR, 'x_scaler_num.joblib')
    PT_AMOUNT_PATH = os.path.join(BASE_DIR, 'amount_power_transformer.joblib')
    PT_TARGET_PATH = os.path.join(BASE_DIR, 'target_power_transformer.joblib')
    MODEL_PATH = os.path.join(BASE_DIR, 'XGBoost_Model.joblib')

    if not all([os.path.exists(INPUT_PATH), os.path.exists(LOOKUP_PATH)]):
        print("錯誤：找不到輸入資料表或特徵查找表。")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    print("正在載入模型組件...")
    models = {
        'scaler_x': joblib.load(SCALER_PATH),
        'pt_amount': joblib.load(PT_AMOUNT_PATH) if os.path.exists(PT_AMOUNT_PATH) else None,
        'pt_target': joblib.load(PT_TARGET_PATH),
        'xgb': joblib.load(MODEL_PATH)
    }

    print("正在載入特徵查找表...")
    lookup_df = pd.read_csv(LOOKUP_PATH)
    lookup_df = lookup_df.drop_duplicates(subset=['品號'], keep='first')

    # =========================================================
    # 第一部分：預測製程缺料 (Model缺料物料.csv) -> 產出 缺料風險預測報告.csv
    # =========================================================
    print("【任務 1】正在預測製程缺料...")
    df_all = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    
    mask = df_all['缺料狀態'] == '缺料'
    shortage_df = df_all[mask].copy()
    normal_df = df_all[~mask].copy()

    if not shortage_df.empty:
        # 呼叫預測函式
        pred_days = get_predictions(
            df=shortage_df, 
            lookup_df=lookup_df, 
            id_col='材料品號', 
            name_col='材料品名', 
            spec_col='材料規格', 
            amount_col='缺料數量', 
            models=models
        )
        shortage_df['預測LeadTime'] = pred_days
        
        today = datetime.now()
        shortage_df['預測進貨日期'] = pd.to_datetime(shortage_df['預測LeadTime'].apply(lambda x: today + timedelta(days=int(x))))
        shortage_df['預計開工日'] = pd.to_datetime(shortage_df['預計開工日'])
        
        shortage_df['預計延遲天數'] = (shortage_df['預測進貨日期'] - shortage_df['預計開工日']).dt.days

        conditions = [
            shortage_df['預計延遲天數'] > 5,
            (shortage_df['預計延遲天數'] >= 0) & (shortage_df['預計延遲天數'] <= 5),
            shortage_df['預計延遲天數'] < 0
        ]
        choices = ['極高', '中', '低']
        shortage_df['風險評估'] = np.select(conditions, choices, default='普通')

        shortage_df['預測進貨日期'] = shortage_df['預測進貨日期'].dt.strftime('%Y-%m-%d')
        shortage_df['預計開工日'] = shortage_df['預計開工日'].dt.strftime('%Y-%m-%d')

        shortage_df.rename(columns={
            '預測LeadTime': '預計物料延誤天數',
            '預計開工日': '實際開工日',
            '預計延遲天數': '預計延遲開機天數'
        }, inplace=True)

    if not normal_df.empty:
        normal_df['實際開工日'] = pd.to_datetime(normal_df['預計開工日'], errors='coerce').dt.strftime('%Y-%m-%d')
        for col in ['預計物料延誤天數', '預測進貨日期', '預計延遲開機天數', '風險評估']:
            normal_df[col] = ""

    final_combined = pd.concat([shortage_df, normal_df], ignore_index=True)
    
    output_cols = ['製令單號', '產品品名', '製程名稱', '材料品號', '材料品名', '材料規格', '預計用料', '庫存數量', '缺料數量', '實際開工日', '預計物料延誤天數', '預測進貨日期', '預計延遲開機天數', '風險評估']
    final_combined['預計延遲開機天數'] = final_combined['預計延遲開機天數'].astype(int)
    final_combined['預計物料延誤天數'] = final_combined['預計物料延誤天數'].astype(int)

    for col in output_cols:
        if col not in final_combined.columns:
            final_combined[col] = ""

    final_df = final_combined[output_cols].sort_values(by='實際開工日')

    final_csv = os.path.join(MODEL_DIR, '缺料風險預測報告.csv')
    final_df.to_csv(final_csv, index=False, encoding='utf-8-sig', sep=',')

    output_txt = os.path.join(MODEL_DIR, 'Risk_Report.txt')
    high_risk_df = final_df[final_df['風險評估'].isin(['極高', '中'])]

    with open(output_txt, 'w', encoding='utf-8-sig') as f:
        f.write(f"=== AI 缺料風險預警報告 ({datetime.now().strftime('%Y-%m-%d')}) ===\n\n")
        f.write(f"目前高風險與極高風險物料共 {len(high_risk_df)} 項：\n\n")
        for _, row in high_risk_df.iterrows():
            f.write(f"【製令】: {row['製令單號']}\n")
            f.write(f"  物料: {row['材料品號']} ({row['材料品名']})\n")
            f.write(f"  預計用料: {row['預計用料']}\n")
            f.write(f"  庫存數量: {row['庫存數量']}\n")
            f.write(f"  缺料數量: {row['缺料數量']}\n")
            f.write(f"  實際開工: {row['實際開工日']}\n")
            f.write(f"  預測進貨: {row['預測進貨日期']} (預估: {row['預計物料延誤天數']}天)\n")
            icon = "🚨" if row['風險評估'] == '極高' else "⚠️"
            f.write(f"  狀態: {icon} {row['風險評估']}\n")
            f.write(f"  預計延遲天數: {row['預計延遲開機天數']}\n")
            f.write("-" * 30 + "\n")

    print(f"-> 成功產出 CSV 報表：{final_csv}")
    print(f"-> 成功產出文字報告：{output_txt}")


    # =========================================================
    # 第二部分：預測未交採購單 (PURT.csv) -> 產出 叫料單.csv
    # =========================================================
    # TODO：
    # 把過濾出來以交數量 <=0 的df，生成"未到貨叫料單.csv"
    # 把"未到貨叫料單.csv" 與 "model缺料物料.csv"進行合併。(這裡的資料是已有的叫料訂單與現在缺料的訂單，合併後我想得到的是未來物料數量的波動)
    # 合併方式為: 依照時間排序物料分組，相同物料抓出第一個製程的時間到往後30天的所有製程的所需數量，得到一個各個物料所需要叫料的數量。
    # 把這些數據丟進model 預測。
    # 之後會生成一個"未來叫料單.csv"，裡面會有 [品號,品名,規格,採購數量,預計到料時間]這些欄位。

    print("【任務 2】正在預測採購未交單據 (叫料單)...")
    if os.path.exists(PURT_PATH):
        purt_df = pd.read_csv(PURT_PATH, dtype={'品號': str}, encoding='utf-8-sig')
        
        # 欄位清洗
        purt_df.columns = purt_df.columns.str.strip()
        
        # 確保已交數量為數值
        purt_df['已交數量'] = pd.to_numeric(purt_df['已交數量'], errors='coerce').fillna(0)
        
        # 過濾未到料且符合特定物料前綴的單據
        prefixes = ('M0', 'M2', 'E', 'K', 'm0', 'm2', 'e', 'k')
        unfulfilled_mask = (purt_df['已交數量'] <= 0) & (purt_df['品號'].str.startswith(prefixes))
        purt_target_df = purt_df[unfulfilled_mask].copy()
        
        if not purt_target_df.empty:
            purt_pred_days = get_predictions(
                df=purt_target_df, 
                lookup_df=lookup_df, 
                id_col='品號', 
                name_col='品名', 
                spec_col='規格', 
                amount_col='採購數量', 
                models=models
            )
            purt_target_df['預測LeadTime'] = purt_pred_days
            
            # 計算預計到料時間 = 採購日期 + 預測LeadTime
            # 如果沒有採購日期，則用今天
            purt_target_df['採購日期'] = pd.to_datetime(purt_target_df['採購日期'], errors='coerce')
            default_date = pd.to_datetime(datetime.now().date())
            purt_target_df['採購日期'] = purt_target_df['採購\46
            '日期'].fillna(default_date)
            
            purt_target_df['預計到料時間'] = purt_target_df.apply(
                lambda row: row['採購日期'] + timedelta(days=int(row['預測LeadTime'])), axis=1
            )
            
            purt_target_df['預計到料時間'] = purt_target_df['預計到料時間'].dt.strftime('%Y-%m-%d')
            purt_target_df['採購日期'] = purt_target_df['採購日期'].dt.strftime('%Y-%m-%d')
            
            # 整理叫料單欄位
            order_cols = ['採購單別', '採購單號', '品號', '品名', '規格', '採購數量', '已交數量', '預計到料時間']

            purt_target_df['採購單別'] = purt_target_df['採購單別'].astype(int)
            purt_target_df['採購單號'] = purt_target_df['採購單號'].astype(int)
            purt_target_df['採購數量'] = purt_target_df['採購數量'].astype(int)
            purt_target_df['已交數量'] = purt_target_df['已交數量'].astype(int)

            # 保留有在 DataFrame 的欄位
            available_order_cols = [c for c in order_cols if c in purt_target_df.columns]
            
            final_purt_df = purt_target_df[available_order_cols]
            
            purt_csv = os.path.join(MODEL_DIR, '叫料單預測結果.csv')
            final_purt_df.to_csv(purt_csv, index=False, encoding='utf-8-sig', sep=',')
            print(f"-> 成功產出 叫料單.csv：{purt_csv} (共 {len(final_purt_df)} 筆)")
        else:
            print("-> 無符合條件的未交採購單需要預測。")
    else:
        print(f"警告：找不到 PURT 檔案 {PURT_PATH}")

    print("-" * 30)
    print("所有預測任務執行完畢！")
    print("-" * 30)
    
if __name__ == "__main__":
    prepare_and_predict()