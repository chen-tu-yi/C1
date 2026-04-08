import pandas as pd
import os

def generate_product_master_bom(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"找不到輸入檔案: {input_file}")
        return

    # 1. 讀取原始資料 (確保品號等欄位為字串，避免遺失開頭的 0)
    print(f"正在讀取 {input_file} 的 MOCTA_MOCTB 分頁...")
    df = pd.read_excel(input_file, sheet_name='MOCTA_MOCTB', dtype=str)

    # 2. 數值預處理
    # 將已領用量轉為數字，非數字部分補 0
    qty_col = '已領用量 (TB身)'
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)

    # 3. 定義核心欄位 (對應 tmp 中的名稱)
    col_ta_product = '產品品名 (TA頭)'
    col_ta_order   = '製令單號 (TA頭)'
    col_tb_id      = '材料品號 (TB身)'
    col_tb_name    = '材料品名 (TB身)'
    col_tb_spec    = '材料規格 (TB身)'
    col_tb_wh      = '庫別 (TB身)'

    # 4. 建立「參考來源製令清單」
    # 針對每個 [產品品名]，抓出它所涵蓋的所有唯一 [製令單號]
    print("正在彙整各產品之來源製令...")
    order_groups = df.groupby(col_ta_product)[col_ta_order].unique().apply(lambda x: ', '.join(x)).reset_index()
    order_groups.columns = [col_ta_product, '參考來源製令清單']

    # 5. 核心計算：以產品品名為 key，找出材料需求的最大值
    # 分組邏輯：以產品品名為首，結合材料各項屬性進行分組
    # 這樣可以確保「同一個產品下，相同的材料品名」會在不同單號間取最大值
    print("正在以產品品名為核心計算物料聯集與最大量...")
    # 注意：我們將產品品名放在 index 第一位，滿足您「產品品名為 key」的要求
    df_max_agg = df.groupby(
        [col_ta_product, col_tb_name, col_tb_id, col_tb_spec, col_tb_wh],
        as_index=False
    )[qty_col].max()

    # 6. 合併製令清單資訊
    final_result = pd.merge(df_max_agg, order_groups, on=col_ta_product, how='left')

    # 7. 排序與重新命名輸出欄位
    # 依照產品品名排序，讓同一產品的材料集中在一起
    final_result = final_result.sort_values([col_ta_product, col_tb_name])
    
    final_result.columns = [
        '產品品名', 
        '材料品名', 
        '材料品號', 
        '材料規格', 
        '庫別', 
        '最大需求數量', 
        '參考來源製令清單'
    ]

    # 8. 儲存檔案
    print(f"正在產出分析結果: {output_file}...")
    final_result.to_excel(output_file, index=False)
    print("處理完成！")

# ==========================================
# 執行設定
# ==========================================
INPUT_XLSX = 'tmp.xlsx'
OUTPUT_XLSX = '產品品名物料需求最大化彙整表_修正版.xlsx'

if __name__ == "__main__":
    try:
        generate_product_master_bom(INPUT_XLSX, OUTPUT_XLSX)
    except Exception as e:
        print(f"發生錯誤: {e}")