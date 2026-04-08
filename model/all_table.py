'''
合併model預測缺料與原本沒被抓進去預測的data。輸出為"製令總表.csv"，作為系統面輸出的db。
'''

import pandas as pd
import os

def generate_master_production_report():
    # 1. 設定檔案路徑
    FORECAST_FILE = '缺料風險預測報告.csv'
    INVENTORY_FILE = '製程與所需物料庫存表.csv'
    OUTPUT_FILE = '製令總表.csv'

    # 檢查檔案是否存在
    if not os.path.exists(FORECAST_FILE) or not os.path.exists(INVENTORY_FILE):
        print("錯誤：找不到必要的 CSV 檔案。")
        return

    print("讀取 CSV 資料中...")
    # 讀取資料
    forecast_df = pd.read_csv(FORECAST_FILE)
    inventory_df = pd.read_csv(INVENTORY_FILE)

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

    # 5. 輸出結果
    # 確保輸出格式能讓 Excel 正確分欄
    master_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("製令總表建立完成！")
    print(f"原始物料筆數：{len(inventory_df)}")
    print(f"最終總表筆數：{len(master_df)}")
    print(f"輸出路徑：{os.path.abspath(OUTPUT_FILE)}")
    print("-" * 30)

if __name__ == "__main__":
    generate_master_production_report()