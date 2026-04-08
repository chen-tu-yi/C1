import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 檔案定義
FILENAME = 'C2_v3.xlsx' 
SHEET_DATA = 'Sheet1'
SHEET_HOLIDAY = '節假日表'

def main():
    # 1. 讀取資料
    df = pd.read_excel(FILENAME, sheet_name=SHEET_DATA, header=1) # 標題通常在第二列
    df_holiday = pd.read_excel(FILENAME, sheet_name=SHEET_HOLIDAY, header=None)
    
    # 2. 欄位清洗
    df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
    
    # 3. 執行狀態篩選 (排除 未開始 與 空白)
    # 確保狀態欄位乾淨
    col_status = '製程狀態'
    df[col_status] = df[col_status].astype(str).str.strip()
    
    mask_exclude = (df[col_status] == '未開始') | (df[col_status] == 'nan') | (df[col_status].isna())
    df_filtered = df[~mask_exclude].copy()
    
    # 4. 日期格式轉換
    date_cols = ['預計開工日', '預計完工日', '實際開工日', '實際完工日']
    for col in date_cols:
        df_filtered[col] = pd.to_datetime(df_filtered[col], errors='coerce')

    # 5. 準備節假日清單 (Set 格式加速計算)
    holiday_list = pd.to_datetime(df_holiday.iloc[:, 0], errors='coerce').dt.date.tolist()
    holiday_set = set([h for h in holiday_list if pd.notna(h)])

    # 6. 定義工作日計算函數 (扣除週末與假日)
    def calc_workdays(start, end, holidays):
        if pd.isna(start) or pd.isna(end) or end < start:
            return 0
        all_dates = pd.date_range(start=start, end=end)
        # 排除週末 (weekday < 5) 與 節假日表
        workdays = [d for d in all_dates if d.weekday() < 5 and d.date() not in holidays]
        return len(workdays)

    # 7. 執行計算與判定
    print("正在執行 C2 延遲判定邏輯...")
    
    # A. 物料延遲: 實際開工 > 預計開工
    df_filtered['Is_Material_Delay'] = df_filtered['實際開工日'] > df_filtered['預計開工日']
    df_filtered['物料延遲天數'] = (df_filtered['實際開工日'] - df_filtered['預計開工日']).dt.days

    # B. 人員延遲: (實完-實開) > (預完-預開)
    # 僅針對「已完成」的項目計算工作日
    df_filtered['預計工作日'] = df_filtered.apply(lambda r: calc_workdays(r['預計開工日'], r['預計完工日'], holiday_set), axis=1)
    df_filtered['實際工作日'] = df_filtered.apply(lambda r: calc_workdays(r['實際開工日'], r['實際完工日'], holiday_set), axis=1)
    
    df_filtered['Is_Manpower_Delay'] = (
        (df_filtered[col_status] == '已完成') & 
        (df_filtered['實際工作日'] > df_filtered['預計工作日'])
    )

    # 8. 儲存結果
    df_filtered.to_excel('C2_ana_filtered_final.xlsx', index=False)
    print(f"清洗完成！保留筆數: {len(df_filtered)}")

if __name__ == "__main__":
    main()