import pandas as pd
import numpy as np
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 設定檔案名稱
FILE_DATA = 'C2_ana_filtered.xlsx'    # 要計算的資料檔 (過濾後的)
FILE_HOLIDAY = 'C2_v3.xlsx'           # 節假日來源檔 (請確認此檔案有 '節假日表' sheet)
SHEET_DATA_NAME = 'Sheet1'            # 資料 Sheet 名稱 (Pandas 預設輸出通常是 Sheet1)
SHEET_HOLIDAY_NAME = '節假日表'       # 假日 Sheet 名稱

def main():
    print(f"1. Reading Data from {FILE_DATA}...")
    
    # 讀取資料表
    try:
        # 先嘗試 header=0 (通常 pandas 輸出的 excel標題在第一列)
        df = pd.read_excel(FILE_DATA, sheet_name=SHEET_DATA_NAME, header=0)
    except Exception as e:
        print(f"❌ 讀取資料失敗: {e}")
        return

    # 讀取節假日表
    print(f"2. Reading Holidays from {FILE_HOLIDAY}...")
    try:
        # 這裡假設假日表在 FILE_HOLIDAY 裡
        df_holidays = pd.read_excel(FILE_HOLIDAY, sheet_name=SHEET_HOLIDAY_NAME, header=None)
        
        # 處理假日資料: 轉為 datetime -> 取 date -> 轉 set
        holiday_list = pd.to_datetime(df_holidays.iloc[:, 0], errors='coerce').dt.date.tolist()
        holiday_set = set([h for h in holiday_list if pd.notna(h)])
        
        print(f"✅ 成功讀取 {len(holiday_set)} 筆節假日資料")
        
    except Exception as e:
        print(f"⚠️ 讀取節假日失敗 ({e})，將只扣除週末，不扣除國定假日。")
        holiday_set = set()

    # 定義欄位名稱 (請確認這些欄位在您的 excel 中存在)
    # 如果您的欄位名稱不同 (例如有 '派工資訊' 字樣)，請在此修改
    col_act_start = '實際開工日' 
    col_act_end = '實際完工日'
    
    # 自動搜尋欄位 (怕欄位名稱有微小差異)
    if col_act_start not in df.columns:
        col_act_start = next((c for c in df.columns if '實際' in c and '開工' in c), col_act_start)
    if col_act_end not in df.columns:
        col_act_end = next((c for c in df.columns if '實際' in c and '完工' in c), col_act_end)

    print(f"   - 使用欄位: {col_act_start} (開始), {col_act_end} (結束)")

    # 3. 定義計算函數
    def calculate_workdays(start_date, end_date, holidays):
        # 檢查是否為空值
        if pd.isna(start_date) or pd.isna(end_date):
            return 0  # 或回傳 np.nan
        
        # 確保格式正確
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 若 結束 < 開始，視為 0
        if end < start:
            return 0

        # A. 產生日期序列 (包含開始與結束)
        all_dates = pd.date_range(start=start, end=end)
        
        # B. 排除週末 (Saturday=5, Sunday=6)
        workdays = all_dates[all_dates.weekday < 5]
        
        # C. 排除國定假日
        actual_workdays = [d for d in workdays if d.date() not in holidays]
        
        return len(actual_workdays)

    # 4. 執行計算
    print("3. Calculating Actual Workdays...")
    
    new_col_name = '實際工作日天數'
    
    # 使用 apply 逐列計算
    df[new_col_name] = df.apply(
        lambda row: calculate_workdays(row[col_act_start], row[col_act_end], holiday_set), 
        axis=1
    )

    # 5. 調整欄位順序 (將新欄位移到最後)
    # 雖然剛新增的欄位預設就在最後，但為了保險起見可以重排
    cols = [c for c in df.columns if c != new_col_name] + [new_col_name]
    df = df[cols]

    print("   - 計算完成！前 5 筆預覽:")
    print(df[[col_act_start, col_act_end, new_col_name]].head())

    # 6. 儲存結果
    OUTPUT_FILE = 'C2_ana_result.xlsx'
    print(f"4. Saving to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    print("🎉 Done!")

if __name__ == "__main__":
    main()