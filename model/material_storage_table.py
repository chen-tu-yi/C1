'''
抓取2025與MOCTA_MOCTB或許製程與物料對應表單。
整理成一張製令與所需材料及其庫存的表單
'''


import pandas as pd
import os
import re


def find_header_and_read(file_path, sheet_name, keyword='製令單號'):
    """
    自動尋找標題列位置並讀取 Excel (處理標題在第二列的問題)。
    """
    preview = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5, header=None)
    header_idx = 0
    for i, row in preview.iterrows():
        if row.astype(str).str.contains(keyword).any():
            header_idx = i
            break
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    df.columns = [re.sub(r'[\s\n.]+', '', str(c)) for c in df.columns]
    return df



def fetch_demand_data():
    """
    從 000-2025模組進線日期.xlsx 獲取需求與預計開工日。
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '000-2025模組進線日期.xlsx')
    plan_df = find_header_and_read(file_path, '整機計劃')
    mask = (plan_df['製令工單生產狀態'] == '未生產') & (plan_df['執行狀態'] == '未開始')
    filtered = plan_df[mask].copy()
    return filtered[['製令單號', '規格', '製程名稱', '預計開工日']]

def fetch_bom_data():
    """
    從 ERP_Table.xlsx MOCTA_MOCTB 獲取產品與製程的 BOM 對照表。
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ERP_Table.xlsx')
    # 加入 dtype={...} 強制指定型態
    bom_df = pd.read_excel(file_path, sheet_name='MOCTA_MOCTB', dtype={'製令單別 (TB身)': str, '製令單號 (TB身)': str})
    bom_df.columns = bom_df.columns.str.strip()
    
    # 新增過濾邏輯：[品號]欄位只看開頭為 M0, M2, E, K 的物料
    prefixes = ('M0', 'M2', 'E', 'K', 'm0', 'm2', 'e', 'k')
    bom_df = bom_df[bom_df['材料品號 (TB身)'].astype(str).str.startswith(prefixes)].copy()
    
    return bom_df[['製令單別 (TB身)', '製令單號 (TB身)', '產品規格 (TA頭)', '材料品號 (TB身)', '需領用量 (TB身)', '材料品名 (TB身)', '材料規格 (TB身)']]

def fetch_inventory_data():
    """
    從 ERP_Table.xlsx INVMB 獲取即時庫存數量。
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ERP_Table.xlsx')
    inventory_data = pd.read_excel(file_path, sheet_name='INVMB')
    inventory_data.columns = inventory_data.columns.str.strip()
    return inventory_data[['品號 (MB)', '庫存數量 (MB)']]

def calculate_material_shortage(demand_df, bom_df, inventory_data):
    """
    1. 將需求單與 BOM 展開結合以製令單號合併
    2. 結合庫存狀態以材料品號合併
    3. 依據時間線推算累計需求，標記出「缺料」狀態
    """
    # 定義一個清洗函數，把浮點數生成的 .0 殺掉
    def clean_id(s):
        return str(s).replace('.0', '').strip()

    # 處理計畫表的單號
    demand_df['製令單號'] = demand_df['製令單號'].apply(clean_id)
    
    # 處理 ERP BOM 的單號
    bom_df['製令單別 (TB身)'] = bom_df['製令單別 (TB身)'].apply(clean_id)
    bom_df['製令單號 (TB身)'] = bom_df['製令單號 (TB身)'].apply(clean_id)
    
    # 重新合併單號
    bom_df['合併製令單號'] = bom_df['製令單別 (TB身)'] + '-' + bom_df['製令單號 (TB身)']

    bom_df['材料品號 (TB身)'] = bom_df['材料品號 (TB身)'].astype(str).str.strip()
    inventory_data['品號 (MB)'] = inventory_data['品號 (MB)'].astype(str).str.strip()

    # 3. BOM 展開 (工單與 BOM 關聯)
    model_data = pd.merge(
        demand_df, 
        bom_df, 
        left_on='製令單號', 
        right_on='合併製令單號', 
        how='left'
    )
    
    # 5. 合併庫存
    final_model = pd.merge(
        model_data,
        inventory_data,
        left_on='材料品號 (TB身)',
        right_on='品號 (MB)',
        how='left'
    )
    
    # 6. 計算與標記邏輯
    final_model['庫存數量 (MB)'] = final_model['庫存數量 (MB)'].fillna(0)
    
    # 確保日期欄位為日期格式
    final_model['預計開工日'] = pd.to_datetime(final_model['預計開工日'], errors='coerce')
    final_model.sort_values(by=['預計開工日', '製令單號'], inplace=True)
    
    # 針對每一種材料計算隨時間增加的累計需求
    final_model['累計需求'] = final_model.groupby('材料品號 (TB身)')['需領用量 (TB身)'].cumsum()
    
    final_model['缺料狀態'] = final_model.apply(
        lambda x: '缺料' if x['累計需求'] > x['庫存數量 (MB)'] else '庫存充足', 
        axis=1
    )
    
    # 將數值欄位轉成整數，確保無小數點殘留
    final_model['需領用量 (TB身)'] = pd.to_numeric(final_model['需領用量 (TB身)'], errors='coerce').fillna(0).astype(int)
    final_model['累計需求'] = pd.to_numeric(final_model['累計需求'], errors='coerce').fillna(0).astype(int)
    final_model['庫存數量 (MB)'] = pd.to_numeric(final_model['庫存數量 (MB)'], errors='coerce').fillna(0).astype(int)
    
    # 計算缺料數量，並將小於 0 的值直接 clip 為 0
    final_model['缺料數量'] = (final_model['累計需求'] - final_model['庫存數量 (MB)']).clip(lower=0).astype(int)

    # 取代欄位名稱
    rename_rules = {
        '需領用量 (TB身)': '預計用料',
        '庫存數量 (MB)': '庫存數量',
        '材料品號 (TB身)': '材料品號',
        '材料品名 (TB身)': '材料品名',
        '材料規格 (TB身)': '材料規格',
        '規格': '產品品名' # 依照原本設計使用 demand_df 裡的規格，命名為產品品名
    }
    final_model.rename(columns=rename_rules, inplace=True)

    # 整理輸出欄位
    final_output_cols = [
        '製令單號', '預計開工日', '產品品名', '製程名稱', 
        '材料品號', '材料品名', '材料規格', 
        '預計用料', '累計需求', '庫存數量', '缺料狀態', '缺料數量'
    ]
    return final_model[final_output_cols]

def run_shortage_model():
    print("正在獲取來源數據...")
    demand_df = fetch_demand_data()
    print(f"-> 計畫表過濾後剩下: {len(demand_df)} 筆") # 診斷點 1
    
    if len(demand_df) > 0:
        print(f"   範例單號: {demand_df['製令單號'].iloc[0]}")

    bom_df = fetch_bom_data()
    print(f"-> ERP BOM 原始資料: {len(bom_df)} 筆") # 診斷點 2

    # 強制修正：確保 BOM 合併鍵與計畫表一致
    # 有些 ERP 單號需要補齊長度，這裡我們假設是字串直接串接
    bom_df['製令單別 (TB身)'] = bom_df['製令單別 (TB身)'].astype(str).str.strip()
    bom_df['製令單號 (TB身)'] = bom_df['製令單號 (TB身)'].astype(str).str.strip()
    bom_df['合併製令單號'] = bom_df['製令單別 (TB身)'] + "-" + bom_df['製令單號 (TB身)']
    
    if len(bom_df) > 0:
        print(f"   ERP 合併後範例單號: {bom_df['合併製令單號'].iloc[0]}")

    inventory_data = fetch_inventory_data()
    
    print("開始執行 Join 與計算...")
    # 執行合併前，再次確保 key 乾淨
    demand_df['製令單號'] = demand_df['製令單號'].astype(str).str.strip()
    
    # 進行合併
    result_df = calculate_material_shortage(demand_df, bom_df, inventory_data)

    # 9. 輸出完整分析結果
    file_path_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '製程與所需物料庫存表.csv')
    result_df.to_csv(file_path_csv, index=False, encoding='utf-8-sig')

    # 10. 過濾並輸出「缺料物料清單」
    short_result_df = result_df[result_df['缺料狀態'] == '缺料'].copy()
    file_path_short = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model缺料物料.csv')
    short_result_df.to_csv(file_path_short, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"完整需求資料：{len(result_df)} 筆 (製程與所需物料庫存表.csv)")
    print(f"僅缺料項目：{len(short_result_df)} 筆 (model缺料物料.csv)")
    print("-" * 30)

if __name__ == "__main__":
    run_shortage_model()