'''
合併model預測缺料與原本沒被抓進去預測的data。輸出為"製令總表.csv"，作為系統面輸出的db。
'''

import pandas as pd
import os

def generate_master_production_report():
    MODEL_DIR = r'C:\local_file\專題\model'
    
    INVENTORY_FILE = os.path.join(MODEL_DIR, '製程與所需物料庫存表.csv')
    PREDICTED_FILE = os.path.join(MODEL_DIR, 'Model缺料物料_加上預測與標記.csv')
    OUTPUT_FILE = os.path.join(MODEL_DIR, '製令總表.csv')

    if not os.path.exists(INVENTORY_FILE) or not os.path.exists(PREDICTED_FILE):
        print("錯誤：找不到必要的 CSV 檔案。")
        return

    print("讀取 CSV 資料中...")
    inv_df = pd.read_csv(INVENTORY_FILE, encoding='utf-8-sig')
    pred_df = pd.read_csv(PREDICTED_FILE, encoding='utf-8-sig')

    # 清洗合併鍵，避免空白或型態差異
    join_keys = ['製令單號', '產品品名', '製程名稱', '材料品號', '材料品名', '材料規格']
    
    for key in join_keys:
        if key in inv_df.columns:
            inv_df[key] = inv_df[key].astype(str).str.strip()
        if key in pred_df.columns:
            pred_df[key] = pred_df[key].astype(str).str.strip()

    print("正在進行資料合併...")
    
    # 找出 pred_df 新增的欄位
    new_cols = [c for c in pred_df.columns if c not in inv_df.columns]
    
    # 取出合併需要的欄位
    pred_subset = pred_df[join_keys + new_cols].drop_duplicates(subset=join_keys, keep='first')
    
    # Left join: 以全部物料(inv_df)為主體
    master_df = pd.merge(inv_df, pred_subset, on=join_keys, how='left')

    # 處理「缺料狀態標記」欄位：只有在 inv_df 裡（沒被抓去預測、非缺料）的會是 NaN，直接補上 'green'
    if '缺料狀態標記' in master_df.columns:
        master_df['缺料狀態標記'] = master_df['缺料狀態標記'].fillna('green')

    master_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("製令總表建立完成！")
    print(f"原始物料筆數：{len(inv_df)}")
    print(f"最終總表筆數：{len(master_df)}")
    print(f"輸出路徑：{os.path.abspath(OUTPUT_FILE)}")
    print("-" * 30)

if __name__ == "__main__":
    generate_master_production_report()