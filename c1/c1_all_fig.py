import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform
import warnings
import os

# 確保輸出目錄存在
png_dir = r"C:\local_file\專題\png\c1"
os.makedirs(png_dir, exist_ok=True)


# 1. 環境設定
warnings.filterwarnings('ignore')

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

# 輔助函式：箱形圖繪製
def plot_box(df_source, target_index, col_item, title, fname, color_palette):
    if target_index.empty: return
    plot_data = df_source[df_source[col_item].isin(target_index)].copy()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=plot_data, x='Analyzed_Delay', y=col_item, order=target_index, palette=color_palette)
    plt.axvline(0, color='blue', linestyle='--')
    plt.title(title, fontsize=14)
    plt.xlabel('延遲天數', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, fname), dpi=300)
    plt.close()

def main():
    print("正在產出分析圖表...")

    stats_file = "C1_Stats_Filtered.csv"
    df_file = "C1_advance-filter.csv"

    if not os.path.exists(stats_file) or not os.path.exists(df_file):
        print(f"找不到所需檔案: {stats_file} 或 {df_file}，請先執行 c1_filter.py 產出資料。")
        return

    try:
        # 讀取 c1_filter.py 產出的 CSV 檔案供 c1_fig.py 分出的畫圖邏輯使用
        stats = pd.read_csv(stats_file, index_col=0)
        df_filtered = pd.read_csv(df_file)
    except Exception as e:
        print(f"讀取 CSV 失敗: {e}")
        return

    col_item = '品名'

    # ---------------------------------------------------------
    # 來自 c1_fig.py 的 11 張圖表產出
    # ---------------------------------------------------------
    # (1) 直方圖 (延遲天數分佈)
    plt.figure(figsize=(12, 6))
    sns.histplot(df_filtered['Analyzed_Delay'], discrete=True, color='#3498db', shrink=0.8)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    plt.title('C1 延遲天數分佈圖', fontsize=14)
    plt.xlabel('延遲天數'), plt.ylabel('筆數'), plt.xlim(-30, 30)
    plt.tight_layout(), plt.savefig(os.path.join(png_dir, 'C1_1_Distribution_advance_filter.png'), dpi=300), plt.close()

    # (2) 圓餅圖 (達交狀況)
    status_counts = df_filtered['Analyzed_Delay'].apply(lambda x: '遲到' if x>0 else ('提早' if x<0 else '準時')).value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(status_counts, labels=[f"{idx}\n{val}筆" for idx, val in zip(status_counts.index, status_counts.values)], 
            autopct='%1.1f%%', colors=[{'遲到':'#e74c3c','提早':'#2ecc71','準時':'#f1c40f'}[x] for x in status_counts.index], startangle=90)
    plt.title('C1 達交狀況圓餅圖-清洗後')
    plt.savefig(os.path.join(png_dir, 'C1_2_Pie_advance_filter.png'), dpi=300), plt.close()
    # 匯出另一張"C1 達交狀態圓餅圖-清洗前"
    try:
        df_removed = pd.read_csv('C1_Removed_Items_Full.csv')
        df_all = pd.concat([df_filtered, df_removed], ignore_index=True)
        status_counts_all = df_all['Analyzed_Delay'].apply(lambda x: '遲到' if x>0 else ('提早' if x<0 else '準時')).value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(status_counts_all, labels=[f"{idx}\n{val}筆" for idx, val in zip(status_counts_all.index, status_counts_all.values)], 
                autopct='%1.1f%%', colors=[{'遲到':'#e74c3c','提早':'#2ecc71','準時':'#f1c40f'}[x] for x in status_counts_all.index], startangle=90)
        plt.title('C1 達交狀況圓餅圖-清洗前')
        plt.savefig(os.path.join(png_dir, 'C1_2_Pie_before_filter.png'), dpi=300), plt.close()
    except Exception as e:
        print(f"無法產生清洗前圓餅圖 (可能是缺少 C1_Removed_Items_Full.csv): {e}")

    # (3) 前15大平均延遲
    top15_mean = stats[stats['mean'] > 0].sort_values('mean', ascending=False).head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='mean', y=top15_mean.index, data=top15_mean, palette='Reds_r')
    plt.title('C1 前十五大平均延遲物料'), plt.tight_layout(), plt.savefig(os.path.join(png_dir, 'C1_3_Top15_AvgDelay_advance_filter.png'), dpi=300), plt.close()

    # (4) 風險地圖
    plt.figure(figsize=(12, 9))
    # 畫出所有物料點作為背景 (灰色)
    sns.scatterplot(data=stats, x='mean', y='std', color='gray', alpha=0.3, s=30)
    
    # 剔除 mean>0 / <0 分頭渲染的限制，直接標註距離原點最遠的前 15 大極端物料 (不分遲到或提前)
    hi_risk = stats.sort_values('origin_dist', ascending=False).head(15)
    sns.scatterplot(data=hi_risk, x='mean', y='std', color='red', s=100)
    for idx, row in hi_risk.iterrows(): 
        plt.text(row['mean'], row['std'], str(idx), fontsize=9, fontweight='bold', color='darkred')

    # 輔助線與標籤設定
    plt.axvline(0, color='blue', linestyle='--', alpha=0.5)
    plt.title('C1 物料風險地圖', fontsize=14)
    plt.xlabel('平均延遲天數 (Mean)'), plt.ylabel('變異標準差 (Std)')
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, 'C1_4_RiskMap_advance_filter.png'), dpi=300)
    plt.close()

    # (5-8) Box Plots (變異分佈)
    # 限定平均延遲大於 1 天，避免選到 mean 接近 0 但 std 巨大的品項
    delayed_idx = stats[stats['mean'] > 1].sort_values('origin_dist', ascending=False).head(10).index
    plot_box(df_filtered, delayed_idx, col_item, '有延遲外圍10點 變異分佈', 'C1_5_Box_Delayed_filter.png', 'Reds')
    
    # 設定 box_forward_filter (提前的品料，mean < -1)
    early_idx = stats[stats['mean'] < -1].sort_values('origin_dist', ascending=False).head(10).index
    plot_box(df_filtered, early_idx, col_item, '提前外圍10點 變異分佈', 'C1_6_Box_Early_filter.png', 'Greens')
    
    # 刪除 box_center_filter (C1_7)
    
    # 保留高變異
    plot_box(df_filtered, stats.sort_values('std', ascending=False).head(15).index, col_item, '前十五大高變異物料 變異分佈', 'C1_8_Box_Top15Std_filter.png', 'Purples')

    # (9) 嚴重遲到 Debug
    top15_severe = stats.sort_values('mean', ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='mean', y=top15_severe.index, data=top15_severe, palette='Reds_r')
    plt.title('C1 十五大嚴重遲到總時間物料'), plt.tight_layout(), plt.savefig(os.path.join(png_dir, 'C1_9_Debug_Severe_filter.png'), dpi=300), plt.close()

    # (10) 延遲占比 - 沒遲前15名 (提早貢獻)
    df_no_delay_f1 = stats[stats['mean'] <= 0].sort_values('total_day', ascending=True).head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_day', y=df_no_delay_f1.index, data=df_no_delay_f1, palette='Greens_r')
    plt.title('延遲占比 - 提早前15 (總提早貢獻)', fontsize=16)
    plt.tight_layout(), plt.savefig(os.path.join(png_dir, 'C1_10_TotalDelay_Negative_Advance_filter.png'), dpi=300), plt.close()

    # (11) 延遲占比 - 遲到前15名 (遲到貢獻)
    df_delayed_f1 = stats[stats['mean'] > 0].sort_values('total_day', ascending=False).head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_day', y=df_delayed_f1.index, data=df_delayed_f1, palette='Reds_r')
    plt.title('延遲占比 - 遲到前15 (總延遲貢獻)', fontsize=16)
    plt.tight_layout(), plt.savefig(os.path.join(png_dir, 'C1_11_TotalDelay_Positive_Advance_filter.png'), dpi=300), plt.close()


    try:
        df = pd.read_csv(stats_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Normalize columns just in case
    df.columns = [str(c).strip().replace('\n', '') for c in df.columns]
    
    # Identify columns
    col_item_2 = next((c for c in df.columns if '品名' in c), '品名')
    col_mean_2 = next((c for c in df.columns if 'mean' in c), 'mean')
    col_count_2 = next((c for c in df.columns if 'count' in c), 'count')

    # Calculate total_day
    df['total_day'] = df[col_mean_2] * df[col_count_2]

    # Split into two groups
    df_delayed_2 = df[df[col_mean_2] > 0].copy()
    df_no_delay_2 = df[df[col_mean_2] <= 0].copy()

    #  12: Delayed Group (Top 15 by total_day) ---
    top15_delayed = df_delayed_2.sort_values('total_day', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_day', y=col_item_2, data=top15_delayed, palette='Reds_r')
    plt.title('延遲占比 - 遲到前15名 -  ', fontsize=16)
    plt.xlabel('總延遲天數', fontsize=12)
    plt.ylabel('品名', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, 'C1_11_TotalDelay_Positive_Advance_filter.png'), dpi=300)
    print("Chart 1 saved: C1_TotalDelay_Positive.png")

    # 13: No Delay Group (Top 15 by total_day) ---
    top15_no_delay = df_no_delay_2.sort_values('total_day', ascending=True).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='total_day', y=col_item_2, data=top15_no_delay, palette='Greens_r')
    plt.title('延遲占比 - 沒遲前15名', fontsize=16)
    plt.xlabel('總提早天數', fontsize=12)
    plt.ylabel('品名', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, 'C1_10_TotalDelay_Negative_Advance_filter.png'), dpi=300)
    print("Chart 2 saved: C1_TotalDelay_Negative.png")


    print("圖像產出全部完成。")

if __name__ == "__main__":
    main()
