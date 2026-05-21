'''
計算資料的MA那些特徵，最後輸出模型所需輸入的特徵
再每次模型預測前執行，更新各個物料的數值
'''

import pandas as pd
import numpy as np
import os

from material_filters import TARGET_ITEM_PREFIXES, filter_to_p_items, filter_to_target_prefixes


# 產生模型推論時使用的物料歷史特徵查找表。
def generate_model_features():
    # 1. 設定絕對路徑
    MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(MODEL_DIR)
    BASE_DIR = os.path.join(PROJECT_DIR, 'c1')
    ADV_PATH = os.path.join(BASE_DIR, 'C1_advance_filter.csv')
    STATS_PATH = os.path.join(BASE_DIR, 'C1_Stats_Filtered.csv')
    OUTPUT_PATH = os.path.join(MODEL_DIR, 'C1_Inference_Features_Lookup.csv')
    
    if not os.path.exists(ADV_PATH):
        print(f"錯誤：找不到輸入檔案 {ADV_PATH}")
        return

    print("正在讀取採購交易數據...")
    # 讀取交易檔
    df = pd.read_csv(ADV_PATH, dtype={'品號': str})
    
    # 欄位清洗與格式轉換
    df.columns = df.columns.str.strip()

    # 先保留原本品號前綴範圍，再只保留 INVMB 品號屬性為 P 的物料。
    before_prefix_count = len(df)
    df = filter_to_target_prefixes(df, '品號')
    print(
        "完成品號前綴過濾 "
        f"(prefixes={TARGET_ITEM_PREFIXES}，{before_prefix_count} -> {len(df)} 筆)"
    )

    erp_path = os.path.join(PROJECT_DIR, 'ERP_Table.xlsx')
    try:
        before_p_count = len(df)
        df, valid_items = filter_to_p_items(df, '品號', erp_path=erp_path)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"錯誤：無法套用 INVMB 品號屬性 P 過濾，停止產出。原因：{e}")
        return

    print(
        "完成品號屬性過濾 "
        f"(INVMB 品號屬性 = 'P'，P 品號數 {len(valid_items)}，"
        f"{before_p_count} -> {len(df)} 筆)"
    )

    # 確保關鍵欄位轉為數值
    col_leadtime = '實際進貨天數' if '實際進貨天數' in df.columns else '進貨天數'
    num_cols = [col_leadtime, '預計進貨天數', '採購數量', '延遲天數']
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
        return group[col_leadtime].tail(n).mean()

    # 4. 彙整統計數據 (Grouping by 品號)
    # 包含 CONFIG_FEATURES 裡設定的所有數值項
    stats_df = df.groupby('品號').agg(
        品名=('品名', 'first'),
        規格=('規格', 'first'),
        # LeadTime_Hist
        Hist_Mean_LeadTime=(col_leadtime, 'mean'),
        Hist_Std_LeadTime=(col_leadtime, 'std'),
        Hist_Min_LeadTime=(col_leadtime, 'min'),
        Hist_Max_LeadTime=(col_leadtime, 'max'),
        Hist_Purchase_Count=(col_leadtime, 'count'),
        Hist_Purchase_Amount=('採購數量', 'sum'),
        # Expected_LeadTime (取中位數作為該品號的標稱預計天數)
        Expected_LeadTime_Median=('預計進貨天數', 'median'),
        # Amount (取中位數作為典型採購量)
        Typical_Amount=('採購數量', 'median')
    ).reset_index()

    # 5. 計算 LeadTime_Actual_Hist (Actual stats & CV)
    actual_stats = df.groupby('品號').agg(
        Hist_Actual_Mean=(col_leadtime, 'mean'),
        Hist_Actual_Std=(col_leadtime, 'std'),
        Hist_Actual_Min=(col_leadtime, 'min'),
        Hist_Actual_Max=(col_leadtime, 'max')
    ).reset_index()
    
    stats_df = pd.merge(stats_df, actual_stats, on='品號', how='left')
    
    stats_df['Hist_Std_LeadTime'] = stats_df['Hist_Std_LeadTime'].fillna(0)
    stats_df['Hist_Actual_Std'] = stats_df['Hist_Actual_Std'].fillna(0)
    
    # CV = 標準差 / 平均值 (衡量穩定度)
    stats_df['Hist_Actual_CV'] = stats_df['Hist_Actual_Std'] / stats_df['Hist_Actual_Mean'].replace(0, 1)

    # 6. 計算 LeadTime_MA (MA3, MA5)
    ma3 = df.groupby('品號')[col_leadtime].apply(lambda s: s.tail(3).mean()).to_dict()
    ma5 = df.groupby('品號')[col_leadtime].apply(lambda s: s.tail(5).mean()).to_dict()    
    
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
