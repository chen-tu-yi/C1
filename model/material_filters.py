"""
共用物料過濾工具。

所有模型推論串接若需要限制物料範圍，統一使用原本品號前綴範圍與 INVMB 的品號屬性 P。
"""

import os

import pandas as pd


P_ITEM_PROPERTY = "P"
INVMB_ITEM_COL = "品號 (MB)"
INVMB_PROPERTY_COL = "品號屬性 (MB)"
TARGET_ITEM_PREFIXES = ("M0", "M2", "E", "K", "m0", "m2", "e", "k")


# 取得專案根目錄的 ERP_Table.xlsx 預設路徑。
def get_default_erp_path():
    model_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(model_dir, "..", "ERP_Table.xlsx")


# 將品號欄位統一轉成可比對的字串格式。
def normalize_item_series(series):
    return series.astype(str).str.strip()


# 從 INVMB 讀取品號屬性為 P 的品號集合。
def load_p_item_ids(erp_path=None):
    erp_path = erp_path or get_default_erp_path()

    if not os.path.exists(erp_path):
        raise FileNotFoundError(f"找不到 ERP_Table.xlsx：{erp_path}")

    required_cols = {INVMB_ITEM_COL, INVMB_PROPERTY_COL}
    invmb_df = pd.read_excel(
        erp_path,
        sheet_name="INVMB",
        dtype=str,
        usecols=lambda col: str(col).strip() in required_cols,
    )
    invmb_df.columns = invmb_df.columns.astype(str).str.strip()

    missing_cols = required_cols - set(invmb_df.columns)
    if missing_cols:
        missing_text = ", ".join(sorted(missing_cols))
        raise ValueError(f"INVMB 缺少必要欄位：{missing_text}")

    property_mask = (
        invmb_df[INVMB_PROPERTY_COL]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(P_ITEM_PROPERTY)
    )

    item_ids = normalize_item_series(invmb_df.loc[property_mask, INVMB_ITEM_COL])
    item_ids = item_ids[item_ids.ne("") & item_ids.str.lower().ne("nan")]

    return set(item_ids)


# 用已載入的 P 品號集合過濾指定 DataFrame。
def filter_to_item_ids(df, item_col, item_ids):
    if item_col not in df.columns:
        raise KeyError(f"資料缺少品號欄位：{item_col}")

    filtered = df.copy()
    filtered[item_col] = normalize_item_series(filtered[item_col])
    return filtered[filtered[item_col].isin(item_ids)].copy()


# 用原本的 M0/M2/E/K 品號前綴範圍過濾指定 DataFrame。
def filter_to_target_prefixes(df, item_col, prefixes=TARGET_ITEM_PREFIXES):
    if item_col not in df.columns:
        raise KeyError(f"資料缺少品號欄位：{item_col}")

    filtered = df.copy()
    filtered[item_col] = normalize_item_series(filtered[item_col])
    return filtered[filtered[item_col].str.startswith(prefixes)].copy()


# 讀取 INVMB P 品號集合後過濾指定 DataFrame。
def filter_to_p_items(df, item_col, erp_path=None):
    item_ids = load_p_item_ids(erp_path)
    return filter_to_item_ids(df, item_col, item_ids), item_ids
