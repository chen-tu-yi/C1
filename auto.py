import pandas as pd
import pyodbc
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 設定區
# ==========================================
DB_CONFIG = {
    'SERVER': 'localhost\\SQLEXPRESS',      
    'DATABASE': 'VISIONWIDE',   
    'DRIVER': '{ODBC Driver 17 for SQL Server}',
    'TRUSTED': 'yes'
}

TABLE_NAME_FILE = 'ERP_Table_Name.xlsx'       
OUTPUT_FILE = 'ERP_Table.xlsx'  # 最終產出的單一 Excel 檔案
# ==========================================

def deduplicate_names(names):
    """防止 Excel 欄位名稱重複導致寫入失敗"""
    seen = {}
    result = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            result.append(name)
        else:
            seen[name] += 1
            result.append(f"{name}.{seen[name]}")
    return result

def get_prefix(table_name):
    if not table_name: return "XX"
    return table_name[-2:]

def load_whitelist_and_targets():
    """直接讀取清單並分析合併目標"""
    if not os.path.exists(TABLE_NAME_FILE):
        print(f"找不到 {TABLE_NAME_FILE}")
        return None, []
    
    try:
        df_input = pd.read_excel(TABLE_NAME_FILE, engine='openpyxl')
        df_white = pd.DataFrame()
        df_white['檔案代號'] = df_input.iloc[:, 0].astype(str).str.strip().str.upper()
        df_white['欄位代號'] = df_input.iloc[:, 1].astype(str).str.strip().str.upper()
        df_white['中文名稱'] = df_input.iloc[:, 2].astype(str).str.strip()
        df_white = df_white[df_white['檔案代號'] != 'NAN']

        unique_tables = set(df_white['檔案代號'].unique())
        targets = []
        processed = set()

        # 1. 強制指定的配對關係 (包含 MOCTD + MOCTE)
        explicit_pairs = [
            ('MOCTA', 'MOCTB'), ('MOCTC', 'MOCTE'),
            ('PURTG', 'PURTH'), ('BOMME', 'BOMMF'), ('BOMMC', 'BOMMD'),
            ('SFCTD', 'SFCTE'), ('SFCTE', 'SFCTF'), ('PURTC', 'PURTD')
        ]
        
        for head, body in explicit_pairs:
            if head in unique_tables and body in unique_tables and head not in processed and body not in processed:
                targets.append((head, body, f"{head}_{body}"))
                processed.update([head, body])

        # 2. 自動探測其餘連續字母配對 (A/B, C/D...)
        for table in sorted(list(unique_tables)):
            if table in processed: continue
            suffix = table[-1]
            for h, b in [('A','B'), ('C','D'), ('E','F'), ('G','H')]:
                if suffix == h:
                    body = table[:-1] + b
                    if body in unique_tables and body not in processed:
                        targets.append((table, body, f"{table}_{body}"))
                        processed.update([table, body])
                        break
            if table not in processed:
                targets.append((table, None, table))
                processed.add(table)
        
        return df_white, targets
    except Exception as e:
        print(f"讀取失敗: {e}")
        return None, []

def get_join_condition(table_a, table_b):
    pre_a, pre_b = get_prefix(table_a), get_prefix(table_b)
    if table_a == 'BOMMC' and table_b == 'BOMMD':
        return f"A.{pre_a}001=B.{pre_b}001"
    return f"A.{pre_a}001=B.{pre_b}001 AND A.{pre_a}002=B.{pre_b}002"

def fetch_data_from_sql(target):
    conn_str = (f"DRIVER={DB_CONFIG['DRIVER']};SERVER={DB_CONFIG['SERVER']};"
                f"DATABASE={DB_CONFIG['DATABASE']};Trusted_Connection={DB_CONFIG['TRUSTED']};")
    table_a, table_b, base_name = target
    if table_b:
        join_on = get_join_condition(table_a, table_b)
        sql_query = f"SELECT A.*, B.* FROM {table_a} AS A LEFT JOIN {table_b} AS B ON {join_on}"
    else:
        sql_query = f"SELECT * FROM {table_a}"
    try:
        with pyodbc.connect(conn_str) as conn:
            return pd.read_sql(sql_query, conn)
    except Exception as e:
        print(f"抓取失敗 ({base_name}): {e}")
        return None

def process_data(df_raw, whitelist_map, target):
    if df_raw is None or df_raw.empty: return None
    table_a, table_b, base_name = target
    
    new_cols, cols_to_keep = [], []
    for col in df_raw.columns:
        clean_col = col.strip()
        prefix = clean_col[:2]
        
        t = None
        suffix = ""
        if prefix == get_prefix(table_a):
            t = table_a
            suffix = "頭" if table_b else ""
        elif table_b and prefix == get_prefix(table_b):
            t = table_b
            suffix = "身"
            
        if t and (t, clean_col) in whitelist_map:
            cols_to_keep.append(col)
            new_cols.append(f"{whitelist_map[(t, clean_col)]} ({prefix}{suffix})")
            
    if not cols_to_keep: return None
    df_clean = df_raw[cols_to_keep]
    df_clean.columns = deduplicate_names(new_cols)
    return df_clean.replace(r'^\s*$', None, regex=True)

def main():
    df_white, targets = load_whitelist_and_targets()
    if df_white is None: return

    whitelist_map = {(str(r['檔案代號']).upper(), str(r['欄位代號']).upper()): str(r['中文名稱']) 
                     for _, r in df_white.iterrows()}

    print(f"開始寫入總表: {OUTPUT_FILE}")
    sheets_written = 0
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for target in targets:
            sheet_name = target[2][:31] # Excel 分頁名稱上限 31 字
            print(f"正在處理分頁: {sheet_name}...", end=" ")
            
            df_raw = fetch_data_from_sql(target)
            df_final = process_data(df_raw, whitelist_map, target)
            
            if df_final is not None and not df_final.empty:
                df_final.to_excel(writer, sheet_name=sheet_name, index=False)
                sheets_written += 1
                print("成功")
            else:
                print("跳過")

    if sheets_written > 0:
        print(f"✨ 全部合併完成！請查看資料夾中的 {OUTPUT_FILE}")
    else:
        print("❌ 未產出任何資料，請檢查 SQL 連線或欄位清單。")

if __name__ == "__main__":
    main()