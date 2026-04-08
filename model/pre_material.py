'''
計算資料的MA那些特徵，最後輸出模型所需輸入的特徵
再每此模型預測前執行，更新各個物料的數值
'''

import pandas as pd
import numpy as np
import os

def generate_model_features():
    # 1. 設定絕對路徑
    BASE_DIR = r'C:\local_file\專題\c1'
    ADV_PATH = os.path.join(BASE_DIR, 'C1_advance_filter.csv')
    STATS_PATH = os.path.join(BASE_DIR, 'C1_Stats_Filtered.csv')
    OUTPUT_PATH = 'C1_Inference_Features_Lookup.csv'
    
    if not os.path.exists(ADV_PATH):
        print(f"錯誤：找不到輸入檔案 {ADV_PATH}")
        return

    print("正在讀取採購交易數據...")
    # 讀取交易檔
    df = pd.read_csv(ADV_PATH, dtype={'品號': str})
    
    # 欄位清洗與格式轉換
    df.columns = df.columns.str.strip()

    # --- 新增過濾邏輯：只看品號開頭為 M0, M2, E, K 的物料 ---
    # 使用 tuple 傳入 startswith 可一次過濾多個前綴 (不分大小寫處理)
    prefixes = ('M0', 'M2', 'E', 'K', 'm0', 'm2', 'e', 'k')
    df = df[df['品號'].str.startswith(prefixes)].copy()
    print(f"完成品號過濾，賸餘資料筆數：{len(df)}")


    # 確保關鍵欄位轉為數值
    num_cols = ['進貨天數', '預計進貨天數', '採購數量', '延遲天數']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 2. 依照時間排序 (為了計算正確的 MA)
    # 假設採購日期欄位存在，若無則按原始索引排序
    date_col = '採購日期' if '採購日期' in df.columns else '單據日期'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=[date_col, '品號'])

    print("開始計算模型所需特徵 (MA3, MA5, Hist Stats)...")

    # 3. 定義 MA 計算函式 (針對品號的最新 N 筆)
    def get_latest_ma(group, n):
        return group['進貨天數'].tail(n).mean()

    # 4. 彙整統計數據 (Grouping by 品號)
    # 包含 CONFIG_FEATURES 裡設定的所有數值項
    stats_df = df.groupby('品號').agg(
        品名=('品名', 'first'),
        規格=('規格', 'first'),
        # LeadTime_Hist
        Hist_Mean_LeadTime=('進貨天數', 'mean'),
        Hist_Std_LeadTime=('進貨天數', 'std'),
        Hist_Min_LeadTime=('進貨天數', 'min'),
        Hist_Max_LeadTime=('進貨天數', 'max'),
        Hist_Purchase_Count=('進貨天數', 'count'),
        Hist_Purchase_Amount=('採購數量', 'sum'),
        # Expected_LeadTime (取中位數作為該品號的標稱預計天數)
        Expected_LeadTime_Median=('預計進貨天數', 'median'),
        # Amount (取中位數作為典型採購量)
        Typical_Amount=('採購數量', 'median')
    ).reset_index()

    # 5. 計算 LeadTime_Actual_Hist (Actual stats & CV)
    col_actual = '實際進貨天數' if '實際進貨天數' in df.columns else '進貨天數'
    actual_stats = df.groupby('品號').agg(
        Hist_Actual_Mean=(col_actual, 'mean'),
        Hist_Actual_Std=(col_actual, 'std'),
        Hist_Actual_Min=(col_actual, 'min'),
        Hist_Actual_Max=(col_actual, 'max')
    ).reset_index()
    
    stats_df = pd.merge(stats_df, actual_stats, on='品號', how='left')
    
    stats_df['Hist_Std_LeadTime'] = stats_df['Hist_Std_LeadTime'].fillna(0)
    stats_df['Hist_Actual_Std'] = stats_df['Hist_Actual_Std'].fillna(0)
    
    # CV = 標準差 / 平均值 (衡量穩定度)
    stats_df['Hist_Actual_CV'] = stats_df['Hist_Actual_Std'] / stats_df['Hist_Actual_Mean'].replace(0, 1)

    # 6. 計算 LeadTime_MA (MA3, MA5)
    ma3 = df.groupby('品號').apply(lambda x: get_latest_ma(x, 3)).to_dict()
    ma5 = df.groupby('品號').apply(lambda x: get_latest_ma(x, 5)).to_dict()
    
    stats_df['LeadTime_MA3'] = stats_df['品號'].map(ma3)
    stats_df['LeadTime_MA5'] = stats_df['品號'].map(ma5)

    # 7. 處理對齊 c1_pre.py 的轉換特徵
    # Log 轉換與 PT 轉換在 pre_model 處理，此處保留原始值
    stats_df['Expected_LeadTime_Log_Base'] = np.log1p(stats_df['Expected_LeadTime_Median'].clip(lower=0))
    
    # 建立 Hash 特徵所需字串
    stats_df['Name_Spec'] = stats_df['品名'].astype(str) + "_" + stats_df['規格'].astype(str)

    # 8. 輸出 CSV
    stats_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print("特徵數據彙整完成！")
    print(f"產出檔案：{OUTPUT_PATH}")
    print(f"總計品號數量：{len(stats_df)}")
    print("-" * 30)

if __name__ == "__main__":
    generate_model_features()