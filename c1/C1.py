import pandas as pd
import os
import warnings

# 忽略警告訊息
warnings.filterwarnings('ignore')

# ==========================================
# 設定檔案名稱
# ==========================================
INPUT_EXCEL = "Final_ERP_Report_Important.xlsx"
OUTPUT_EXCEL = "Procurement_Precise_Analysis_v13.xlsx"

# ==========================================
# 1. 強力清洗函式 (解決對不上的關鍵)
# ==========================================

def clean_col_name(col):
    """
    清洗欄位名稱
    移除 Excel 匯出時的括號與說明，例如 '單別 (TH頭)' -> '單別'
    """
    return str(col).split(' ')[0].split('(')[0]

def clean_key_robust(series):
    """
    [一般鍵值清洗] 用於單別、單號
    功能：強制轉字串，去除 '.0' (浮點數轉整數)，去除空白
    解決：'20241001.0' 對不上 '20241001' 的問題
    """
    if isinstance(series, pd.DataFrame): 
        series = series.iloc[:, 0]
    
    # 先轉字串，用 split 去掉小數點後的東西，再 strip 去空白
    return series.astype(str).str.split('.').str[0].str.strip()

def clean_seq_robust(series):
    """
    [序號專用清洗] 用於採購序號
    功能：去除 '.0' 後，強制補零至 4 位數
    解決：'1' 對不上 '0001' 的問題
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    # 1. 轉字串並去小數點 (處理 1.0)
    s = series.astype(str).str.split('.').str[0].str.strip()
    
    # 2. 強制補零到 4 位 (處理 1 -> 0001)
    # 注意：如果原本就是 '0001'，zfill(4) 不會改變它
    return s.str.zfill(4)

def parse_date(series):
    """日期解析：將 YYYYMMDD 字串轉為 datetime"""
    if isinstance(series, pd.DataFrame): 
        series = series.iloc[:, 0]
    s_str = series.astype(str).str.split('.').str[0].str.strip()
    return pd.to_datetime(s_str, format='%Y%m%d', errors='coerce')

def load_sheet_by_keyword(xls, keyword):
    """搜尋並讀取 Excel 分頁"""
    sheet_name = next((s for s in xls.sheet_names if keyword in s), None)
    if sheet_name:
        print(f"      ✅ 讀取分頁: {sheet_name}")
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [clean_col_name(c) for c in df.columns]
            return df
        except Exception as e:
            print(f"      ❌ 讀取失敗: {e}")
            return pd.DataFrame()
    else:
        print(f"      ⚠️ 找不到包含 '{keyword}' 的分頁")
        return pd.DataFrame()

# ==========================================
# 2. 主程式邏輯
# ==========================================

def main():
    print("🚀 執行：採購流程精確對齊分析 (v13 - 強力格式整平版)")
    print(f"📂 來源檔案: {INPUT_EXCEL}")
    print("="*50)

    if not os.path.exists(INPUT_EXCEL):
        print(f"❌ 錯誤：找不到檔案 '{INPUT_EXCEL}'")
        return

    try:
        xls = pd.ExcelFile(INPUT_EXCEL)
    except Exception as e:
        print(f"❌ 無法開啟 Excel 檔案: {e}")
        return

    # 1. 讀取資料
    # ------------------------------------------------
    print("\n1️⃣  讀取資料中...")
    df_th = load_sheet_by_keyword(xls, "PURTH")       # 驗收明細 (主角)
    df_tg = load_sheet_by_keyword(xls, "PURTG")       # 進貨單頭 (提供進貨日期)
    df_pur = load_sheet_by_keyword(xls, "PURTC_PURTD") # 採購單 (提供採購日期/交期)
    
    if df_pur.empty:
        print("      ⚠️ 嘗試尋找 PURTC 單獨分頁...")
        df_pur = load_sheet_by_keyword(xls, "PURTC")

    if df_th.empty:
        print("❌ 錯誤：缺少核心資料 PURTH，程式終止。")
        return

    # 2. 建立強力關聯鍵 (The Clean Keys)
    # ------------------------------------------------
    print("\n2️⃣  建立強力關聯鍵 (修正 1.0 vs 0001 問題)...")

    # --- A. 處理 TH (驗收單) ---
    # 這是要連去 TG 的鑰匙
    df_th['Link_TG_Type'] = clean_key_robust(df_th['單別'])
    df_th['Link_TG_No'] = clean_key_robust(df_th['單號'])

    # 這是要連去 PUR 的鑰匙 (來源)
    df_th['Link_PO_Type'] = clean_key_robust(df_th['採購單別'])
    df_th['Link_PO_No'] = clean_key_robust(df_th['採購單號'])
    
    # 序號處理 (最關鍵的一步！)
    seq_col = next((c for c in ['採購序號', '序號'] if c in df_th.columns), None)
    if seq_col:
        df_th['Link_PO_Seq'] = clean_seq_robust(df_th[seq_col])
    else:
        # 若真的沒有序號欄位，補 '0000' 防止程式崩潰
        df_th['Link_PO_Seq'] = '0000'

    # TH 自己的日期
    df_th['DT_驗收'] = parse_date(df_th['驗收日期']) if '驗收日期' in df_th.columns else pd.NaT

    # --- B. 處理 TG (進貨單) ---
    if not df_tg.empty:
        df_tg['Link_TG_Type'] = clean_key_robust(df_tg['單別'])
        df_tg['Link_TG_No'] = clean_key_robust(df_tg['單號'])
        
        # 準備日期
        if '進貨日期' in df_tg.columns: df_tg['DT_進貨'] = parse_date(df_tg['進貨日期'])
        else: df_tg['DT_進貨'] = pd.NaT
            
        if '單據日期' in df_tg.columns: df_tg['DT_TG單據'] = parse_date(df_tg['單據日期'])
        else: df_tg['DT_TG單據'] = pd.NaT

    # --- C. 處理 PUR (採購單) ---
    if not df_pur.empty:
        # 判斷欄位名稱
        p_type = '採購單別' if '採購單別' in df_pur.columns else '單別'
        p_no = '採購單號' if '採購單號' in df_pur.columns else '單號'
        
        df_pur['Link_PO_Type'] = clean_key_robust(df_pur[p_type])
        df_pur['Link_PO_No'] = clean_key_robust(df_pur[p_no])
        
        # 序號處理 (同樣補零到 4 位)
        p_seq = next((c for c in ['序號', '採購序號'] if c in df_pur.columns), None)
        if p_seq:
            df_pur['Link_PO_Seq'] = clean_seq_robust(df_pur[p_seq])
        else:
            df_pur['Link_PO_Seq'] = '0000'
            
        # 日期準備
        if '採購日期' in df_pur.columns: df_pur['DT_採購'] = parse_date(df_pur['採購日期'])
        else: df_pur['DT_採購'] = pd.NaT
        
        d_due = parse_date(df_pur['預交日']) if '預交日' in df_pur.columns else pd.NaT
        d_cfm = parse_date(df_pur['交期確認日']) if '交期確認日' in df_pur.columns else pd.NaT
        
        # 優先使用交期確認日，若無則用預交日
        df_pur['DT_目標交期'] = d_cfm.fillna(d_due)

    # 3. 資料串接 (Merging)
    # ------------------------------------------------
    print("\n3️⃣  開始串接資料...")
    
    # 步驟 1: TH 找 TG (補 進貨日期)
    df_merged = df_th.copy()
    if not df_tg.empty:
        df_merged = pd.merge(df_merged, 
                           df_tg[['Link_TG_Type', 'Link_TG_No', 'DT_進貨', 'DT_TG單據']], 
                           on=['Link_TG_Type', 'Link_TG_No'], 
                           how='left')
    else:
        df_merged['DT_進貨'] = pd.NaT
        df_merged['DT_TG單據'] = pd.NaT

    # 步驟 2: 結果 找 PUR (補 採購日期 & 預交日)
    # 先刪除可能存在的衝突欄位
    df_merged = df_merged.drop(columns=['DT_採購', 'DT_目標交期'], errors='ignore')
    
    df_final = df_merged.copy()
    if not df_pur.empty:
        # 使用 Type + No + Seq 三鍵值精確對接
        df_final = pd.merge(df_final, 
                          df_pur[['Link_PO_Type', 'Link_PO_No', 'Link_PO_Seq', 'DT_採購', 'DT_目標交期']], 
                          on=['Link_PO_Type', 'Link_PO_No', 'Link_PO_Seq'], 
                          how='left')
    else:
        df_final['DT_採購'] = pd.NaT
        df_final['DT_目標交期'] = pd.NaT

    # 4. 補位計算與延遲判定
    # ------------------------------------------------
    print("\n4️⃣  計算補位與延遲...")

    # 確保所有欄位都存在 (防呆)
    for col in ['DT_採購', 'DT_TG單據', 'DT_進貨', 'DT_驗收', 'DT_目標交期']:
        if col not in df_final.columns: df_final[col] = pd.NaT

    # A. [開始日]: 優先用採購日期，若無則用 TG單據日期
    df_final['開始日'] = df_final['DT_採購'].fillna(df_final['DT_TG單據'])

    # B. [結束日]: 優先用進貨日期，若無則用 驗收日期
    df_final['結束日'] = df_final['DT_進貨'].fillna(df_final['DT_驗收'])

    # C. [延遲天數]: 目標交期 - 結束日
    df_final['延遲天數'] = (df_final['DT_目標交期'] - df_final['結束日']).dt.days

    # 5. 格式化輸出
    # ------------------------------------------------
    # 日期轉字串
    date_cols = ['開始日', '結束日', 'DT_目標交期', 'DT_採購', 'DT_TG單據', 'DT_進貨', 'DT_驗收']
    for col in date_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].dt.strftime('%Y-%m-%d')

    # 欄位中文化重命名 (讓報表好讀)
    rename_map = {
        'DT_目標交期': '預計交期(來源:PUR)',
        'DT_採購': '採購日期(來源:PUR)',
        'DT_TG單據': 'TG單據日期(來源:TG)',
        'DT_進貨': '進貨日期(來源:TG)',
        'DT_驗收': '驗收日期(來源:TH)'
    }
    df_final.rename(columns=rename_map, inplace=True)

    # 決定輸出欄位
    # 識別欄位
    cols_id = ['單別', '單號', '品號', '品名', '規格', '驗收數量', '單位']
    # 計算欄位
    cols_calc = ['開始日', '結束日', '延遲天數']
    # 來源欄位 (保留下來讓您核對)
    cols_source = ['採購單別', '採購單號', '採購序號', '採購日期(來源:PUR)', 'TG單據日期(來源:TG)', '預計交期(來源:PUR)', '進貨日期(來源:TG)', '驗收日期(來源:TH)']
    
    final_cols = []
    for c in cols_id + cols_calc + cols_source:
        if c in df_final.columns:
            final_cols.append(c)

    # 排序
    sort_cols = [c for c in ['單別', '單號', '品號'] if c in df_final.columns]
    if sort_cols:
        df_final.sort_values(by=sort_cols, inplace=True)

    # 寫入 Excel
    print(f"\n5️⃣  寫入檔案: {OUTPUT_EXCEL}")
    try:
        df_final[final_cols].to_excel(OUTPUT_EXCEL, index=False)
        print("\n" + "="*50)
        print(f"🎉 分析完成！")
        print(f"📄 檔案已儲存為: {OUTPUT_EXCEL}")
        print(f"📊 總筆數: {len(df_final)}")
        
        # 統計成功率
        matched_po = df_final['採購日期(來源:PUR)'].notnull().sum()
        print(f"   🔗 成功關聯到採購單 (有採購日期): {matched_po} 筆 ({matched_po/len(df_final):.1%})")
        print("      (若此數字仍低，請確認 PURTC_PURTD 是否包含對應年份的資料)")
        print("="*50)
    except Exception as e:
        print(f"❌ 存檔失敗: {e}")

if __name__ == "__main__":
    main()