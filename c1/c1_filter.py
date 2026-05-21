'''
過濾PURTG/H 表單，需更改為從ERP_Table.xlsx中抓取sheet-PURTG/H，而分C1.xlsx的表格。(目前的C1是由PURTC/D/G/H共同組成)
最後advanced_filter.csv，供c1_pre.py做前處理使用
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform
import warnings
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from material_filters import TARGET_ITEM_PREFIXES, filter_to_p_items, filter_to_target_prefixes

# 1. 環境設定
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 字型設定 (支援繁體中文)
system_name = platform.system()
if system_name == "Windows":
    font_list = ['Microsoft JhengHei', 'Microsoft JhengHei UI', 'PMingLiU', 'SimHei', 'Arial']
elif system_name == "Darwin":
    font_list = ['Heiti TC', 'PingFang TC', 'Arial']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Arial']

plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.sans-serif": font_list})

# 清理圖表與輸出使用的品名標籤。
# 輔助函式：清理標籤字串
def clean_label(text):
    if isinstance(text, str):
        return text.replace('(', '').replace(')', '').replace('（', '').replace('）', '').strip()
    return text

# 繪製目前資料範圍的物料風險散佈圖。
def plot_riskmap(df, col_item, col_delay, step_name):
    # 計算每個物料的平均值和標準差，dropna 是為了忽略只有單筆紀錄導致無法計算 std 的物料
    png_dir = r"C:\local_file\專題\png\c1"
    stats = df.groupby(col_item)[col_delay].agg(['mean', 'std']).dropna()
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=stats, x='mean', y='std', color='red', s=100)
    plt.axvline(0, color='blue', linestyle='--', alpha=0.5)
    plt.title(f'Material Risk Map - {step_name}', fontsize=24, fontweight='bold')
    plt.xlabel('Mean Delay Days', fontsize=20)
    plt.ylabel('Standard Deviation of Delay', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-50, 50)
    plt.ylim(0, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, f"RiskMap_{step_name}.png"), dpi=300)
    plt.close()

# 執行 C1 採購資料清洗與特徵統計產出流程。
def main():
    file_path = "PURT.csv"
    print(f"開始執行 C1 自動化分析流程，讀取檔案: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"找不到檔案: {file_path}")
        return

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig') 
    except Exception as e:
        print(f"讀取 CSV 失敗: {e}")  
        return

    # ---------------------------------------------------------
    # 2. 基礎清洗與欄位對應
    # ---------------------------------------------------------
    df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
    
    col_delay = next((c for c in df.columns if '延遲天數' in c), '延遲天數')
    col_item = next((c for c in df.columns if '品名' in c), '品名')
    col_actual = next((c for c in df.columns if '進貨日期' in c), None)
    col_expect = next((c for c in df.columns if '預交日' in c), None)
    col_id = next((c for c in df.columns if '品號' in c), None)
    col_elt = next((c for c in df.columns if '預計進貨天數' in c), None)
    col_amount = next((c for c in df.columns if '採購數量' in c), None)
    col_leadtime = next((c for c in df.columns if c in ['進貨天數(已扣除假日)', '實際進貨天數', '進貨天數']), '進貨天數(已扣除假日)')
    
    # 抓取未扣除假日之進貨天數 (可能欄位名稱包含"進貨天數"但不包含"扣除", 或名為"總進貨天數")
    # 嚴謹起見，直接以包含"進貨天數"且不包含"扣除"來捕捉，或若使用者有明確的名稱可以直接填寫。
    # 產出時會將此欄位強制更名為 "進貨天數(包含非工作日)"
    clo_leadtime_all = next((c for c in df.columns if '進貨天數' in c and '扣除' not in c and '預計' not in c and c != col_leadtime), None)


    df_clean = df.dropna(subset=[col_delay, col_item]).copy()
    if clo_leadtime_all:
        df_clean[clo_leadtime_all] = pd.to_numeric(df_clean[clo_leadtime_all], errors='coerce')
        # 統一輸出命名為 "進貨天數(包含非工作日)"
        df_clean = df_clean.rename(columns={clo_leadtime_all: '進貨天數(包含非工作日)'})
        clo_leadtime_all = '進貨天數(包含非工作日)'

    df_clean[col_leadtime] = pd.to_numeric(df_clean[col_leadtime], errors='coerce')
    df_clean = df_clean.rename(columns={col_leadtime: '實際進貨天數'})
    col_leadtime = '實際進貨天數'

    if col_elt:
        df_clean[col_elt] = pd.to_numeric(df_clean[col_elt], errors='coerce')
        df_clean[col_delay] = df_clean[col_leadtime] - df_clean[col_elt]

    df_clean[col_delay] = pd.to_numeric(df_clean[col_delay], errors='coerce')
    df_clean = df_clean.dropna(subset=[col_delay])

    df_clean[col_item] = df_clean[col_item].apply(clean_label)
    
    # 產出第一張 Risk map: 最原始、開始過濾前的狀態
    plot_riskmap(df_clean, col_item, col_delay, '0_before_filter')

    # ---------------------------------------------------------
    # 3. 範圍過濾 (Scope & Lead Time Filter)
    # ---------------------------------------------------------
    # (A) 品號範圍篩選，再套用 INVMB 品號屬性 P。
    if not col_id:
        print("錯誤：找不到品號欄位，無法套用品號範圍與 INVMB 品號屬性 P 過濾。")
        return

    before_prefix_count = len(df_clean)
    df_clean = filter_to_target_prefixes(df_clean, col_id)
    print(
        "品號範圍篩選完成 "
        f"(prefixes={TARGET_ITEM_PREFIXES}，{before_prefix_count} -> {len(df_clean)} 筆)"
    )

    erp_path = os.path.join(PROJECT_DIR, 'ERP_Table.xlsx')
    try:
        before_p_count = len(df_clean)
        df_clean, valid_items = filter_to_p_items(df_clean, col_id, erp_path=erp_path)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"錯誤：無法套用 INVMB 品號屬性 P 過濾，停止產出。原因：{e}")
        return

    print(
        "品號屬性篩選完成 "
        f"(INVMB 品號屬性 = 'P'，P 品號數 {len(valid_items)}，"
        f"{before_p_count} -> {len(df_clean)} 筆)"
    )
    plot_riskmap(df_clean, col_item, col_delay, '1_scope_filter')

    # (B) {預計}進貨天數 <= 90 天
    if col_elt:
        df_clean[col_elt] = pd.to_numeric(df_clean[col_elt], errors='coerce')
        df_clean = df_clean[df_clean[col_elt] <= 90].copy()
        print(f"預計進貨天數篩選完成 (<= 90天)")
        plot_riskmap(df_clean, col_item, col_delay, '2_leadtime_filter')

    # ---------------------------------------------------------
    # 4. 雙重統計過濾 (Count >= 3 & IQR)
    # ---------------------------------------------------------
    print("執行雙重統計過濾...")
    n_before_stats = len(df_clean)

    # Step 1: Count >= 3
    item_counts = df_clean[col_item].value_counts()
    valid_items = item_counts[item_counts >= 3].index
    df_step1 = df_clean[df_clean[col_item].isin(valid_items)].copy()
    plot_riskmap(df_step1, col_item, col_delay, '3_count_filter')

    # Step 2: IQR 離群值 (保留 1.5*IQR 內或 ±15天內)
    grouped = df_step1.groupby(col_item)[col_delay]
    Q1 = grouped.transform(lambda x: x.quantile(0.25))
    Q3 = grouped.transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    condition = (
        ((df_step1[col_delay] >= (Q1 - 1.5 * IQR)) & (df_step1[col_delay] <= (Q3 + 1.5 * IQR))) | 
        (df_step1[col_delay].abs() <= 15)
    )
    df_filtered = df_step1[condition].copy()
    print(f"資料清洗報告: 從 {n_before_stats} 筆 降至 {len(df_filtered)} 筆")
    plot_riskmap(df_filtered, col_item, col_delay, '4_iqr_filter')

    # ---------------------------------------------------------
    # 5. 統計特徵計算
    # ---------------------------------------------------------
    # 分別計算 延遲天數 的統計量，以及採購數量的總和
    agg_args = {
        'mean': (col_delay, 'mean'),
        'std': (col_delay, 'std'),
        'count': (col_delay, 'count'),
        'min': (col_delay, 'min'),
        'max': (col_delay, 'max')
    }
    if col_amount:
        agg_args['amount'] = (col_amount, 'sum')

    col_actual_leadtime = next((c for c in df_filtered.columns if '進貨天數' in c), None)
    if col_actual_leadtime:
        agg_args['actual_mean'] = (col_actual_leadtime, 'mean')
        agg_args['actual_std'] = (col_actual_leadtime, 'std')
        agg_args['actual_min'] = (col_actual_leadtime, 'min')
        agg_args['actual_max'] = (col_actual_leadtime, 'max')

    stats = df_filtered.groupby(col_item).agg(**agg_args)
    if not col_amount:
        stats['amount'] = 0

    if col_actual_leadtime:
        # 動態異常特性：實際進貨天數變異係數 (CV = std / mean) 捕捉波動幅度
        stats['actual_cv'] = stats['actual_std'] / stats['actual_mean'].replace(0, np.nan)
        stats['actual_cv'] = stats['actual_cv'].fillna(0)

    stats['origin_dist'] = np.sqrt(stats['mean']**2 + stats['std']**2)
    stats['score'] = stats['mean'] + stats['std']
    
    # 計算延遲占比 (Total Impact)
    stats['total_day'] = stats['mean'] * stats['count']

    # ---------------------------------------------------------
    # 7. 檔案儲存
    # ---------------------------------------------------------
    # 為了穩定與讀寫效能，全面採用 CSV 格式
    stats.to_csv("C1_Stats_Filtered.csv", encoding='utf-8-sig')
    
    # 大量資料建議儲存為 CSV 以避免 openpyxl 產生的 MemoryError
    df_filtered.to_csv("C1_advance_filter.csv", index=False, encoding='utf-8-sig')
    
    # 儲存被剔除的資料
    removed_items = df_clean[~df_clean.index.isin(df_filtered.index)].copy()
    removed_items.to_csv("C1_Removed_Items_Full.csv", index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("任務完成！")
    print("產出檔案：C1_Stats_Filtered.csv, C1_advance_filter.csv, C1_Removed_Items_Full.csv")

if __name__ == "__main__":
    main()
