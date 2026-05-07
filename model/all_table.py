'''
合併model預測缺料與原本沒被抓進去預測的data。輸出為"製令總表.csv"，作為系統面輸出的db。
'''

#TODO: 最後一欄位加上[總計需購入數量]，計算為=目前品料庫存-製令單所需量+叫料單進貨量
# data soruce:
# 目前物料庫存 --> 製程與所需物料庫存表.csv[庫存數量]
# 製令單所需量 --> 製程與所需物料庫存表.csv[預計用料]
# 叫料單進貨量 --> 叫料單.csv[採購數量]

import pandas as pd
import os

def generate_master_production_report():
    # 1. 設定檔案路徑
    FORECAST_FILE = '缺料風險預測報告.csv'
    INVENTORY_FILE = '製程與所需物料庫存表.csv'
    PURT_FILE = '叫料單.csv'
    OUTPUT_FILE = '製令總表.csv'

    # 檢查檔案是否存在
    if not os.path.exists(FORECAST_FILE) or not os.path.exists(INVENTORY_FILE) or not os.path.exists(PURT_FILE):
        print("錯誤：找不到必要的 CSV 檔案。")
        return

    print("讀取 CSV 資料中...")
    # 讀取資料
    forecast_df = pd.read_csv(FORECAST_FILE)
    inventory_df = pd.read_csv(INVENTORY_FILE)
    purt_df = pd.read_csv(PURT_FILE)

    # 2. 欄位清洗與標準化
    join_keys = ['製令單號','產品品名', '製程名稱', '材料品號', '材料品名']
    
    for key in join_keys:
        if key in inventory_df.columns:
            inventory_df[key] = inventory_df[key].astype(str).str.strip()
        if key in forecast_df.columns:
            forecast_df[key] = forecast_df[key].astype(str).str.strip()

    print("正在進行資料合併...")
    
    # 3. 挑選缺料風險預測報告中需要附加的欄位
    # 預測報告沒有原本庫存表的全部欄位，所以保留 key 與欲附加的預測指標欄位
    cols_to_keep = join_keys + ['預計物料延誤天數', '預測進貨日期', '預計延遲開機天數', '風險評估']
    # 確保只選取 forecast_df 中實際存在的欄位
    available_cols = [c for c in cols_to_keep if c in forecast_df.columns]
    forecast_lookup = forecast_df[available_cols].drop_duplicates(subset=join_keys, keep='first')

    # 4. 進行 Left Join (以整個製程與所需物料庫存表為主體)
    # 這樣即便沒有預測的資料（非缺料），也會完整保留下來，有預測的就會接上結果
    master_df = pd.merge(
        inventory_df,
        forecast_lookup,
        on=join_keys,
        how='left'
    )

    # 5. 計算[總計需購入數量]
    print("計算總計需購入數量...")
    # 先過濾或清洗叫料單資料
    if '品號' in purt_df.columns and '採購數量' in purt_df.columns:
        purt_df['品號'] = purt_df['品號'].astype(str).str.strip()
        # 加總每個品號的採購數量 (即叫料單進貨量)
        purt_agg = purt_df.groupby('品號', as_index=False)['採購數量'].sum()
        purt_agg.rename(columns={'品號': '材料品號', '採購數量': '進貨量'}, inplace=True)
        
        master_df = pd.merge(master_df, purt_agg, on='材料品號', how='left')
    else:
        master_df['叫料單進貨量'] = 0

    # 確保數值型態計算正確
    master_df['庫存數量'] = pd.to_numeric(master_df['庫存數量'], errors='coerce').fillna(0)
    master_df['預計用料'] = pd.to_numeric(master_df['預計用料'], errors='coerce').fillna(0)
    master_df['進貨量'] = pd.to_numeric(master_df['進貨量'], errors='coerce').fillna(0)

    # 公式: =目前品料庫存-製令單所需量+叫料單進貨量
    master_df['總計需購入數量'] = master_df['庫存數量'] - master_df['預計用料'] + master_df['進貨量']
    master_df['總計需購入數量'] = master_df['總計需購入數量'].astype(int)

    master_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("製令總表建立完成！")
    print(f"原始物料筆數：{len(inventory_df)}")
    print(f"最終總表筆數：{len(master_df)}")
    print(f"輸出路徑：{os.path.abspath(OUTPUT_FILE)}")
    print("-" * 30)

if __name__ == "__main__":
    generate_master_production_report()