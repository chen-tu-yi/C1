import pandas as pd
import os
import re

def find_header_and_read(file_path, sheet_name, keyword='製令單號'):
    preview = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5, header=None)
    header_idx = 0
    for i, row in preview.iterrows():
        if row.astype(str).str.contains(keyword).any():
            header_idx = i
            break
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_idx)
    df.columns = [re.sub(r'[\s\n.]+', '', str(c)) for c in df.columns]
    return df

def test_run():
    plan_path = 'c:/local_file/專題/000-2025模組進線日期.xlsx'
    erp_path = 'c:/local_file/專題/ERP_Table.xlsx'
    
    demand_df = find_header_and_read(plan_path, '整機計劃')
    mask = (demand_df['製令工單生產狀態'] == '未生產') & (demand_df['執行狀態'] == '未開始')
    demand_df = demand_df[mask].copy()[['製令單號', '規格', '製程名稱', '預計開工日']]
    print("Demand len:", len(demand_df))
    
    bom_df = pd.read_excel(erp_path, sheet_name='MOCTA_MOCTB')
    bom_df.columns = bom_df.columns.str.strip()
    bom_df = bom_df[['製令單號 (TA頭)', '產品規格 (TA頭)', '材料品號 (TB身)', '需領用量 (TB身)', '材料品名 (TB身)', '材料規格 (TB身)']]
    print("BOM len:", len(bom_df))
    
    inv_df = pd.read_excel(erp_path, sheet_name='INVMB')
    inv_df.columns = inv_df.columns.str.strip()
    inv_df = inv_df[['品號 (MB)', '庫存數量 (MB)']]
    print("INV len:", len(inv_df))
    
    model_data = pd.merge(demand_df, bom_df, left_on='製令單號', right_on='製令單號 (TA頭)', how='inner')
    print("Merged BOM len:", len(model_data))
    if len(model_data) > 0:
        print("Merged BOM sample:", model_data.head(1).to_dict('records'))
    
if __name__ == '__main__':
    test_run()
