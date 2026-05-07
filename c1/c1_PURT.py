'''
需要把ERP_Table.xlsx中的PURTC/D, PURTG/H 共同合併，組成與下列格式相同的csv檔案。


詳細方法為: 
ERP_Table.xlsx中的sheet-PURTC_PURTD, sheet-PURTG_PURTH 以 [採購單號]欄位做為key，保留：採購單別,單別,採購單號,序號,單號(進貨單),採購日期,單據日期,品號,品名,規格,採購數量,數量合計,已交數量,驗收數量,預交日,進貨日期,驗收日期
最後加上"進貨天數, 預計進貨天數, 延遲天數"，計算方式為：進貨天數=IF(OR(F2="",P2=""),"",NETWORKDAYS(F2,P2)), 預計進貨天數=IF(OR(F2="",O2=""),"",NETWORKDAYS(F2,O2)), 延遲天數=IF(OR(O2="",P2=""),"",T2-S2)
其餘不需保留，並清洗無資料的欄位。，輸出成PURT.csv檔案。
'''

import pandas as pd
import numpy as np
import os
def calculate_networkdays(start_dates, end_dates):
    """
    計算兩個日期陣列之間的營業日 (扣除周休二日) 天數。
    相當於 Excel 的 NETWORKDAYS。
    """
    # 建立遮罩把 NaT 變成 False
    valid_mask = start_dates.notna() & end_dates.notna()
    
    # 預設把不合法的日期先填上 NaN
    result = np.full(start_dates.shape, np.nan)
    
    if valid_mask.any():
        # numpy.busday_count 參數接受 datetime64[D] 原生格式的 numpy array
        # 但在 pandas 裡做轉換時要用 numpy 原生轉換，或者保留成 dt.date 再交給 busday_count
        start_valid = start_dates[valid_mask].dt.date.values.astype('datetime64[D]')
        end_valid = end_dates[valid_mask].dt.date.values.astype('datetime64[D]')
        
        # busday_count 是算 start <= x < end, Networkdays 是 start <= x <= end, 所以 end_valid 都要加一天
        # 在 numpy datetime64[D] 裡面加一天就是加 np.timedelta64(1, 'D')
        end_valid = end_valid + np.timedelta64(1, 'D')
        
        # 只針對有效的列計算
        result[valid_mask] = np.busday_count(start_valid, end_valid)
        
    return result
def main():
    print("🚀 開始執行合併流程：PURTC/D + PURTG/H")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    erp_file = os.path.join(base_dir, 'ERP_Table.xlsx')
    output_file = os.path.join(base_dir, 'c1', 'PURT.csv')
    print(f"📖 讀取檔案中 ({erp_file}) ...")
    # 確保指定為 string 以免前導 0 跑掉
    df_purtc_d = pd.read_excel(erp_file, sheet_name='PURTC_PURTD', dtype=str)
    df_purtg_h = pd.read_excel(erp_file, sheet_name='PURTG_PURTH', dtype=str)
    print("🔧 清洗空白並過濾無效對應鍵 ...")
    # 將需要作為 key 的欄位 strip 乾淨，避免空白引起對不起來的問題
    df_purtc_d['採購單別 (TC頭)'] = df_purtc_d['採購單別 (TC頭)'].str.strip()
    df_purtc_d['採購單號 (TC頭)'] = df_purtc_d['採購單號 (TC頭)'].str.strip()
    df_purtc_d['序號 (TD身)'] = df_purtc_d['序號 (TD身)'].str.replace('.0', '', regex=False).str.strip()
    
    df_purtg_h['採購單別 (TH身)'] = df_purtg_h['採購單別 (TH身)'].str.strip()
    df_purtg_h['採購單號 (TH身)'] = df_purtg_h['採購單號 (TH身)'].str.strip()
    df_purtg_h['採購序號 (TH身)'] = df_purtg_h['採購序號 (TH身)'].str.replace('.0', '', regex=False).str.strip()
    print("🔗 執行合併 (Merge) 中 ...")
    # 以採購單別、採購單號、序號作為 Join Key, 假設我們要留下有採購又有進貨的紀錄 (Inner / Left 都可以，這裡用 left 或 inner 要視商業邏輯而定)
    # 這裡依照原需求，要把兩張表合併起來，我們採用 Left Join (以進貨單 / PURTG_H 為主，或以 PURTC_D 為主)
    # 通常追蹤進貨延遲，是以 "有進貨的紀錄(TG_TH)" 去撈採購紀錄(TC_TD) 的預交日
    merged_df = pd.merge(
        df_purtg_h,
        df_purtc_d,
        left_on=['採購單別 (TH身)', '採購單號 (TH身)', '採購序號 (TH身)'],
        right_on=['採購單別 (TC頭)', '採購單號 (TC頭)', '序號 (TD身)'],
        how='left' 
    )
    print("🧹 選取與重命名所需欄位 ...")
    # 保留欄位：採購單別,單別,採購單號,序號,單號(進貨單),採購日期,單據日期,品號,品名,規格,採購數量,數量合計,已交數量,驗收數量,預交日,進貨日期,驗收日期,結案碼
    # 注意: 我們需要根據 ERP_Table 的特有結尾對應
    
    # 建立重命名 mapping
    columns_mapping = {
        '採購單別 (TH身)': '採購單別',
        '單別 (TG頭)': '單別',
        '採購單號 (TH身)': '採購單號',
        '採購序號 (TH身)': '序號',
        '單號 (TG頭)': '單號(進貨單)',
        '採購日期 (TC頭)': '採購日期',
        '單據日期 (TG頭)': '單據日期',
        '品號 (TH身)': '品號',
        '品名 (TH身)': '品名',
        '規格 (TH身)': '規格',
        '採購數量 (TD身)': '採購數量',      # 來自 TC_TD
        '數量合計 (TG頭)': '數量合計',
        '已交數量 (TD身)': '已交數量',
        '驗收數量 (TH身)': '驗收數量',
        '預交日 (TD身)': '預交日',          # 來自 TC_TD
        '進貨日期 (TG頭)': '進貨日期',
        '驗收日期 (TH身)': '驗收日期',
    }
    # 過濾出存在於 mapping 中的 key，並擷取這些欄位
    available_cols = {k: v for k, v in columns_mapping.items() if k in merged_df.columns}
    final_df = merged_df[list(available_cols.keys())].rename(columns=available_cols)
    
    print("📅 轉換日期格式與計算天數 ...")
    # 將日期字串轉換為 datetime 物件，如果轉換出錯就變成 NaT
    for col in ['採購日期', '單據日期', '預交日', '進貨日期', '驗收日期']:
        # 先清除 .0，避免 '20241011.0' 被誤判
        final_df[col] = final_df[col].astype(str).str.replace('.0', '', regex=False).str.strip()
        # 強制轉為 datetime
        final_df[col] = pd.to_datetime(final_df[col], format='%Y%m%d', errors='coerce').fillna(pd.to_datetime(final_df[col], errors='coerce'))
    # 計算進貨天數 = NETWORKDAYS(採購日期, 進貨日期)
    final_df['進貨天數'] = calculate_networkdays(final_df['採購日期'], final_df['進貨日期'])
    
    # 計算預計進貨天數 = NETWORKDAYS(採購日期, 預交日)
    final_df['預計進貨天數'] = calculate_networkdays(final_df['採購日期'], final_df['預交日'])
    # 計算延遲天數 = 進貨天數 - 預計進貨天數
    final_df['延遲天數'] = final_df['進貨天數'] - final_df['預計進貨天數']
    # 由於天數可能是 NaN，為了格式好看，先補 N/A 或清空，並把其他正常欄位轉成整數字串
    for col in ['進貨天數', '預計進貨天數', '延遲天數']:
         final_df[col] = final_df[col].apply(lambda x: str(int(x)) if pd.notna(x) else '')
    # 把 datetime 再轉回字串格式 YYYY/M/D，符合原本的 output 預期
    for col in ['採購日期', '單據日期', '預交日', '進貨日期', '驗收日期']:
         final_df[col] = final_df[col].dt.strftime('%Y/%m/%d').fillna('')
    print("🗑️ 清洗無資料的欄位 ...")
    # 清洗掉全空欄位
    final_df = final_df.dropna(how='all', axis=1)
    print(f"💾 輸出檔案至: {output_file}")
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✨ 完成！")
if __name__ == '__main__':
    main()