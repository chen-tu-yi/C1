import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import numpy as np
import os



# 確保輸出目錄存在
png_dir = r"C:\local_file\專題\png\c2"
os.makedirs(png_dir, exist_ok=True)

# 1. 環境設定與字型處理 (支援繁體中文顯示)
system_name = platform.system()
if system_name == "Windows":
    font_list = ['Microsoft JhengHei', 'Arial']
elif system_name == "Darwin":
    font_list = ['Heiti TC', 'Arial']
else:
    font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Arial']

plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid", {"font.sans-serif": font_list})

# 2. 讀取資料
file_path = r'C:\local_file\專題\c2\C2_v3.xlsx'
try:
    df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
    if '製程狀態' not in df_raw.columns:
        df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=1)
except:
    df_raw = pd.read_csv(r'C:\local_file\專題\c2\C2_v3.xlsx - Sheet1.csv')

# 3. 定義延遲類別判定函數
def get_delay_cat(row):
    is_mat = str(row.get('物料延遲_判斷', '否')).strip() == '是'
    is_man = str(row.get('人為延遲_判斷', '否')).strip() == '是'
    
    if is_mat and is_man: return '兩者皆有'
    if is_mat: return '僅物料延遲'
    if is_man: return '僅人為延遲'
    return '無延遲'

# 4. 資料清洗與預處理
df_raw.columns = [str(c).strip().replace('\n', '') for c in df_raw.columns]

col_status = '製程狀態'
col_process = '製程名稱整機計畫'
col_mat_delay_days = '物料延遲_總天數'
col_man_delay_days = '人為延遲_總天數'

# 清洗前：全部資料
df_before = df_raw.copy()

# 清洗後：排除 "未開始" 與 空白狀態 (NaN)
df_raw[col_status] = df_raw[col_status].astype(str).str.strip()
mask_exclude = (df_raw[col_status] == '未開始') | (df_raw[col_status] == 'nan') | (df_raw[col_status].isna())
df_after = df_raw[~mask_exclude].copy()

# 數值轉換
df_after[col_mat_delay_days] = pd.to_numeric(df_after[col_mat_delay_days], errors='coerce').fillna(0)
df_after[col_man_delay_days] = pd.to_numeric(df_after[col_man_delay_days], errors='coerce').fillna(0)

# 將 df_after 作為 df_filtered 供後續分析
df_filtered = df_after.copy()

# ---------------------------------------------------------
# 來自 C2_ba.py：清洗前後對照 (Pie Chart)
# ---------------------------------------------------------
cat_before = df_before.apply(get_delay_cat, axis=1).value_counts()
cat_after = df_after.apply(get_delay_cat, axis=1).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = {'無延遲': '#2ecc71', '僅物料延遲': '#e74c3c', '僅人為延遲': '#3498db', '兩者皆有': '#f1c40f'}

axes[0].pie(cat_before, labels=[f"{k}\n({v}筆)" for k, v in cat_before.items()], 
            autopct='%1.1f%%', startangle=140, 
            colors=[colors.get(x, '#bdc3c7') for x in cat_before.index])
axes[0].set_title(f'C2 清洗前延遲類別佔比\n(總計 {len(df_before)} 筆)', fontsize=14)

axes[1].pie(cat_after, labels=[f"{k}\n({v}筆)" for k, v in cat_after.items()], 
            autopct='%1.1f%%', startangle=140, 
            colors=[colors.get(x, '#bdc3c7') for x in cat_after.index])
axes[1].set_title(f'C2 清洗後延遲類別佔比\n(總計 {len(df_after)} 筆)', fontsize=14)

plt.suptitle('C2 生產進度資料清洗前後對照圖', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(png_dir, 'C2_Cleaning_Comparison_Pie.png'), dpi=300)
plt.close()

print(f"對照圖已生成！ 清洗前總數: {len(df_before)} 筆, 清洗後總數: {len(df_after)} 筆")

# ---------------------------------------------------------
# 從 c2_fig.py 合併的功能
# ---------------------------------------------------------
df_filtered['延遲類別'] = df_filtered.apply(get_delay_cat, axis=1)
cat_counts = df_filtered['延遲類別'].value_counts()

# 圖表一：延遲原因類別佔比 (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(cat_counts, labels=[f"{k}\n({v}筆)" for k, v in cat_counts.items()], 
        autopct='%1.1f%%', startangle=140, colors=[colors.get(x, '#bdc3c7') for x in cat_counts.index])
plt.title('C2 延遲原因類別佔比', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Delay_Category_Pie.png'), dpi=300)
plt.close()

# 數值清洗前後比較的圖表
df_num_before = df_before.copy()
df_num_before[col_mat_delay_days] = pd.to_numeric(df_num_before[col_mat_delay_days], errors='coerce').fillna(0)
df_num_before[col_man_delay_days] = pd.to_numeric(df_num_before[col_man_delay_days], errors='coerce').fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
labels = ['物料延遲', '人為延遲']
x = np.arange(len(labels))
width = 0.35

means_before = [df_num_before[col_mat_delay_days].mean(), df_num_before[col_man_delay_days].mean()]
means_after = [df_filtered[col_mat_delay_days].mean(), df_filtered[col_man_delay_days].mean()]

ax.bar(x - width/2, means_before, width, label='清洗前', color='#bdc3c7')
ax.bar(x + width/2, means_after, width, label='清洗後', color='#3498db')

ax.set_ylabel('平均天數(Mean Days)')
ax.set_title('C2 數值清洗前後平均延遲天數比較', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend()
for i, v in enumerate(means_before):
    ax.text(i - width/2, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', color='#7f8c8d')
for i, v in enumerate(means_after):
    ax.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold', color='#2980b9')

plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Numerical_Cleaning_Comparison.png'), dpi=300)
plt.close()

# 圖表二：前 15 大平均物料延遲清洗前後比較 (Bar Chart)
top_mat = df_filtered.groupby(col_process)[col_mat_delay_days].mean().sort_values(ascending=False).head(10)
top_mat_before = df_num_before.groupby(col_process)[col_mat_delay_days].mean().reindex(top_mat.index).fillna(0)

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(top_mat))
height = 0.35

ax.barh(y_pos - height/2, top_mat_before.values, height, label='清洗前', color='#f5b041')
ax.barh(y_pos + height/2, top_mat.values, height, label='清洗後', color='#e74c3c')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_mat.index)
ax.invert_yaxis()
ax.set_xlabel('平均延遲天數')
ax.set_ylabel('製程名稱')
ax.set_title('C2 前10大平均物料延遲 (清洗前後比較)', fontsize=16)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Top15_Material_Delay_Comparison.png'), dpi=300)
plt.close()

# 圖表三：前 15 大平均人為延遲清洗前後比較 (Bar Chart)
top_man = df_filtered.groupby(col_process)[col_man_delay_days].mean().sort_values(ascending=False).head(15)
top_man_before = df_num_before.groupby(col_process)[col_man_delay_days].mean().reindex(top_man.index).fillna(0)

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(top_man))
height = 0.35

ax.barh(y_pos - height/2, top_man_before.values, height, label='清洗前', color='#85c1e9')
ax.barh(y_pos + height/2, top_man.values, height, label='清洗後', color='#3498db')

ax.set_yticks(y_pos)
ax.set_yticklabels(top_man.index)
ax.invert_yaxis()
ax.set_xlabel('平均延遲天數')
ax.set_ylabel('製程名稱')
ax.set_title('C2 前15大平均人為延遲 (清洗前後比較)', fontsize=16)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Top15_Manpower_Delay_Comparison.png'), dpi=300)
plt.close()

# 圖表四：高變異製程延遲分佈 (Boxplot)
top_var_processes = df_filtered.groupby(col_process)[col_mat_delay_days].std().sort_values(ascending=False).head(10).index
df_box = df_filtered[df_filtered[col_process].isin(top_var_processes)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_box, x=col_mat_delay_days, y=col_process, hue=col_process, palette='Set3', legend=False)
plt.axvline(0, color='red', linestyle='--')
plt.title('C2 高變異製程之物料延遲分佈', fontsize=16)
plt.xlabel('延遲天數')
plt.ylabel('製程名稱')
plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Delay_Distribution_Boxplot.png'), dpi=300)
plt.close()

# 圖表五：C2 物料風險地圖 (Risk Map)
stats_c2 = df_filtered.groupby(col_process)[col_mat_delay_days].agg(['mean', 'std', 'count']).dropna()
stats_c2['origin_dist'] = np.sqrt(stats_c2['mean']**2 + stats_c2['std']**2)

selected_delayed = stats_c2[stats_c2['mean'] > 0].sort_values('origin_dist', ascending=False).head(10)
selected_no_delay = stats_c2[stats_c2['mean'] <= 0].sort_values('origin_dist', ascending=False).head(10)
selected_center = stats_c2.sort_values('origin_dist', ascending=True).head(5)

plt.figure(figsize=(14, 10))
sns.scatterplot(data=stats_c2, x='mean', y='std', color='gray', alpha=0.3, s=40, label='其他製程')
sns.scatterplot(data=selected_delayed, x='mean', y='std', color='#e74c3c', s=120, edgecolor='black', label='高延遲風險')
sns.scatterplot(data=selected_no_delay, x='mean', y='std', color='#2ecc71', s=120, edgecolor='black', label='穩定/提早區')
sns.scatterplot(data=selected_center, x='mean', y='std', color='#f1c40f', s=120, edgecolor='black', label='準時核心')

all_labels = pd.concat([selected_delayed, selected_no_delay, selected_center])
for idx, row in all_labels.iterrows():
    plt.text(row['mean'] + 0.2, row['std'] + 0.2, str(idx), 
             fontsize=10, fontweight='bold', alpha=0.8, verticalalignment='bottom')

plt.axvline(0, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.6)
plt.title('C2 製程物料延遲風險地圖', fontsize=18, fontweight='bold')
plt.xlabel('平均延遲天數 (Mean Delay)', fontsize=14)
plt.ylabel('延遲標準差 (Std Dev - 波動性)', fontsize=14)
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(png_dir, 'C2_Risk_Map_Processes.png'), dpi=300)
plt.close()

print("C2 分析圖表 (含清洗對照與風險地圖) 已產出成功並儲存於指定資料夾！")
