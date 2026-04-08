import pandas as pd
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

FILENAME = '.xlsx'
SHEET_MODULE = '模組計畫總表'

OUTPUT_EXCEL_FILTERED = "C2_ana_filtered.xlsx"
OUTPUT_EXCEL_RAW = "C2_ana_raw.xlsx"

def main():
    print(f"Reading {FILENAME} - {SHEET_MODULE}...")
    
    # 讀取 Excel
    try:
        # 優先嘗試 header=1 (第二列為標題)
        df = pd.read_excel(FILENAME, sheet_name=SHEET_MODULE, header=1)
    except Exception as e:
        print(f"Error reading file: {e}")
        # 讀取失敗時嘗試備案 (header=0)
        try:
            print("Trying header=0...")
            df = pd.read_excel(FILENAME, sheet_name=SHEET_MODULE, header=0)
        except:
            return

    # 1. 欄位名稱標準化 (去除換行、空白)
    df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
    
    print(f"原始資料筆數: {len(df)}")

    # 2. 定義關鍵欄位 (加入自動搜尋功能，避免欄位名稱微小差異導致錯誤)
    def find_col(keywords):
        for col in df.columns:
            if all(k in col for k in keywords):
                return col
        return None

    # 優先使用您指定的欄位名稱，若找不到則自動搜尋
    col_status = '製程狀態預計開工日整機計畫'
    if col_status not in df.columns:
        col_status = find_col(['狀態']) or '狀態'

    col_plan_start = '預計開工日整機計畫'
    if col_plan_start not in df.columns:
        col_plan_start = find_col(['預計', '開工']) or '預計開工日'

    col_act_start = '實際開工日派工資訊'
    if col_act_start not in df.columns:
        col_act_start = find_col(['實際', '開工']) or '實際開工日'

    col_plan_end = '預計完工日整機計畫'
    if col_plan_end not in df.columns:
        col_plan_end = find_col(['預計', '完工']) or '預計完工日'

    col_act_end = '實際完工日派工資訊'
    if col_act_end not in df.columns:
        col_act_end = find_col(['實際', '完工']) or '實際完工日'

    # 顯示對應結果
    print("✅ 欄位對應確認:")
    print(f"   - 狀態欄位: {col_status}")
    print(f"   - 預計開工: {col_plan_start}")
    print(f"   - 實際開工: {col_act_start}")

    # 3. 日期格式轉換 (對所有資料執行)
    date_cols = [col_plan_start, col_act_start, col_plan_end, col_act_end]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 4. 全局計算 (為所有資料加上判斷)
    print("⚡ 正在執行延遲判斷邏輯...")
    
    # 計算工期 (總天數)
    if col_plan_end in df.columns and col_plan_start in df.columns:
        df['預計製程用時'] = (df[col_plan_end] - df[col_plan_start]).dt.days
    
    if col_act_end in df.columns and col_act_start in df.columns:
        df['實際製程用時'] = (df[col_act_end] - df[col_act_start]).dt.days
    
    # A. 物料延遲判斷 (實際開工 > 預計開工)
    df['物料延遲'] = (df[col_act_start] > df[col_plan_start])
    
    # B. 人員延遲判斷 (實際工期 > 預計工期 且 狀態為已完成)
    # 確保狀態欄位乾淨
    df['Status_Clean'] = df[col_status].astype(str).str.strip()
    
    df['人力延遲'] = (
        (df['Status_Clean'] == '已完成') & 
        (df['實際工作日天數'] > df['預計製程天數'])
    )

    # 5. 輸出 Raw Data (含未開始的所有資料)
    print(f"💾 正在儲存原始資料 (含判斷結果): {OUTPUT_EXCEL_RAW}")
    df.to_excel(OUTPUT_EXCEL_RAW, index=False)
    
    # 6. 執行過濾 (修正後的邏輯)
    print("✂️ 正在執行過濾...")
    
    # 邏輯：排除 '未開始' 且 排除 '空白/nan'
    # 也就是：(狀態 != 未開始) AND (狀態 != nan)
    mask_exclude = (
        (df['Status_Clean'] == '未開始') | 
        (df['Status_Clean'] == 'nan') | 
        (df[col_status].isna())
    )
    
    df_filtered = df[~mask_exclude].copy()
    
    # 7. 統計過濾後結果
    count_mat = df_filtered['物料延遲'].sum()
    count_man = df_filtered['人力延遲'].sum()
    count_both = len(df_filtered[df_filtered['物料延遲'] & df_filtered['人力延遲']])
    count_none = len(df_filtered[~df_filtered['物料延遲'] & ~df_filtered['人力延遲']])

    print("-" * 30)
    print(f"📈 [過濾後] 延遲分析結果 (從 {len(df)} 筆 -> 保留 {len(df_filtered)} 筆):")
    print(f"   - 📦 物料延遲 (晚開工): {count_mat} 筆")
    print(f"   - 👷 人員延遲 (做得慢): {count_man} 筆")
    print(f"   - ⚠️ 兩者皆有: {count_both} 筆")
    print(f"   - ✅ 無延遲 (順利): {count_none} 筆")
    print("-" * 30)

    # 8. 輸出 Filtered Data
    print(f"💾 正在儲存過濾後資料: {OUTPUT_EXCEL_FILTERED}")
    df_filtered.to_excel(OUTPUT_EXCEL_FILTERED, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 分析完成！")
    print(f"1. 原始完整檔: {OUTPUT_EXCEL_RAW}")
    print(f"2. 過濾分析檔: {OUTPUT_EXCEL_FILTERED}")

if __name__ == "__main__":
    main()