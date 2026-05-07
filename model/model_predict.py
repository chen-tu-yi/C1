'''
model預測缺料物料延誤時間，輸出報告"缺料風險預測報告.csv"與"叫料單.csv"。
'''
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os
import warnings
from datetime import datetime, timedelta
from sklearn.feature_extraction import FeatureHasher

warnings.filterwarnings('ignore')

def get_predictions(df, lookup_df, id_col, name_col, spec_col, amount_col, models):
    # 只取需要的 lookup_df 欄位進行合併，避免欄位名稱衝突 (例如 '品名' 和 '規格')
    lookup_cols = [
        '品號', 'Expected_LeadTime_Median',
        'LeadTime_MA3', 'LeadTime_MA5',
        'Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count', 
        'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount', 
        'Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min', 
        'Hist_Actual_Max', 'Hist_Actual_CV'
    ]
    lookup_subset = lookup_df[lookup_cols]
    
    # 合併查找表 (獲取歷史特徵)
    df_merged = pd.merge(df, lookup_subset, left_on=id_col, right_on='品號', how='left')
    df_merged.fillna(0, inplace=True)

    # ---------------------------------------------------------
    # 2. 數值特徵處理 (必須與 c1_pre.py 的順序完全一致: 10個特徵)
    # ---------------------------------------------------------
    
    # A. 處理數量 (Amount_PT)
    if models['pt_amount']:
        df_merged['Amount_PT'] = models['pt_amount'].transform(df_merged[[amount_col]].fillna(0)).flatten()
    else:
        df_merged['Amount_PT'] = np.log1p(df_merged[amount_col].fillna(0))

    # B. 處理預計進貨天數 (Expected_LeadTime_Log)
    df_merged['Expected_LeadTime_Log'] = np.log1p(df_merged['Expected_LeadTime_Median'].fillna(0).clip(lower=0))

    all_num_cols = [
        'LeadTime_MA3', 'LeadTime_MA5',       
        'Amount_PT',                          
        'Expected_LeadTime_Log',              
        'Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count', 
        'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount', 
        'Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min', 
        'Hist_Actual_Max', 'Hist_Actual_CV'   
    ]
    
    X_num = df_merged[all_num_cols].fillna(0)
    X_num_scaled = models['scaler_x'].transform(X_num)
    
    # ---------------------------------------------------------
    # 3. 類別與雜湊特徵 (Hashing) - 必須與 c1_pre.py 順序一致
    # ---------------------------------------------------------
    sparse_blocks = []
    
    # (A) Category_OneHot (4 dims)
    id_str_upper = df_merged[id_col].astype(str).str.upper()
    cat_mat = np.column_stack([
        id_str_upper.str.startswith('M0').astype(np.int8),
        id_str_upper.str.startswith('M2').astype(np.int8),
        id_str_upper.str.startswith('K').astype(np.int8),
        id_str_upper.str.startswith('E').astype(np.int8)
    ])
    sparse_blocks.append(sp.csr_matrix(cat_mat))
    
    # (B) Hash_ID_Full (256 dims)
    hasher_id_full = FeatureHasher(n_features=256, input_type='string')
    hash_full_sp = hasher_id_full.transform(df_merged[id_col].astype(str).apply(lambda x: [x]))
    sparse_blocks.append(hash_full_sp)
    
    # (C) Hash_ID_Split (128 * 4 = 512 dims)
    hasher_id_split = FeatureHasher(n_features=128, input_type='string')
    id_str_pad = df_merged[id_col].astype(str).apply(lambda x: x.ljust(14, ' '))
    hash_split_sp = sp.hstack([
        hasher_id_split.transform(id_str_pad.str[0:2].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[2:6].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[6:10].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[10:14].apply(lambda x: [x]))
    ], format='csr')
    sparse_blocks.append(hash_split_sp)
    
    # (D) Hash_Spec (512 dims)
    df_merged['Name_Spec'] = df_merged[name_col].astype(str) + "_" + df_merged[spec_col].astype(str).fillna('')
    hasher_spec = FeatureHasher(n_features=512, input_type='string')
    hash_spec_sp = hasher_spec.transform(df_merged['Name_Spec'].apply(lambda x: [x]))
    sparse_blocks.append(hash_spec_sp)

    # 4. 合併為最終稀疏矩陣
    X_final = sp.hstack([sp.csr_matrix(X_num_scaled)] + sparse_blocks, format='csr')

    # 5. 模型預測
    y_pred_pt = models['xgb'].predict(X_final)
    y_pred_days = models['pt_target'].inverse_transform(y_pred_pt.reshape(-1, 1)).flatten()
    
    # 回傳合理天數
    return np.maximum(np.round(y_pred_days), 1).astype(int)



def predict_leadtimes_batch(material_id, material_name, material_spec, order_qtys, lookup_df, models):
    if not order_qtys:
        return []
    import pandas as pd
    df_temp = pd.DataFrame({
        '品號': [material_id]*len(order_qtys),
        '品名': [material_name]*len(order_qtys),
        '規格': [material_spec]*len(order_qtys),
        '採購數量': order_qtys
    })
    preds = get_predictions(df_temp, lookup_df, '品號', '品名', '規格', '採購數量', models)
    return preds

def find_best_orders_for_material(material_id, material_name, material_spec, demands, supplies, inventory, current_date, lookup_df, models, x_days=30, planning_horizon_days=90):
    from datetime import timedelta
    planned_orders = []
    horizon_date = current_date + timedelta(days=planning_horizon_days)
    
    while True:
        shortage_idx = None
        shortage_date = None
        sim_inv = inventory
        
        all_events = []
        for i, d in enumerate(demands):
            all_events.append({'type': 'demand', 'date': d['date'], 'qty': d['qty'], 'idx': d['idx']})
        for s in supplies:
            all_events.append({'type': 'supply', 'date': s['date'], 'qty': s['qty']})
        for p in planned_orders:
            all_events.append({'type': 'supply', 'date': p['eta'], 'qty': p['order_qty']})
            
        all_events.sort(key=lambda x: (x['date'], 0 if x['type'] == 'supply' else 1))
        
        for e in all_events:
            if e['type'] == 'supply':
                sim_inv += e['qty']
            else:
                sim_inv -= e['qty']
                if sim_inv < 0:
                    shortage_idx = e['idx']
                    shortage_date = e['date']
                    break
                    
        if shortage_idx is None or shortage_date > horizon_date:
            break
            
        # Find index in demands array for shortage_idx
        demand_start_idx = 0
        for i, d in enumerate(demands):
            if d['idx'] == shortage_idx:
                demand_start_idx = i
                break
                
        review_end = shortage_date + timedelta(days=x_days)
        candidate_pool = [
            d for d in demands[demand_start_idx:]
            if d['date'] <= review_end
        ]
        
        candidates = []
        order_qtys = []
        valid_candidates = []
        
        for k in range(len(candidate_pool)):
            covered_events = candidate_pool[:k+1]
            target_date = covered_events[-1]['date']
            
            net_inv_at_target = inventory
            for e in all_events:
                if e['date'] <= target_date:
                    if e['type'] == 'supply':
                        net_inv_at_target += e['qty']
                    elif e['type'] == 'demand':
                        net_inv_at_target -= e['qty']
            
            order_qty = -net_inv_at_target if net_inv_at_target < 0 else 0
            if order_qty > 0:
                order_qtys.append(order_qty)
                valid_candidates.append({
                    "target_date": target_date,
                    "order_qty": order_qty,
                    "earliest_required": covered_events[0]['date']
                })
                
        if valid_candidates:
            lead_times = predict_leadtimes_batch(material_id, material_name, material_spec, order_qtys, lookup_df, models)
            for vc, lt in zip(valid_candidates, lead_times):
                eta = current_date + timedelta(days=int(lt))
                delay_days = max(0, (eta - vc["earliest_required"]).days)
                score = delay_days * 100 + vc["order_qty"] * 0.1
                candidates.append({
                    "material_id": material_id,
                    "material_name": material_name,
                    "material_spec": material_spec,
                    "order_qty": vc["order_qty"],
                    "eta": eta,
                    "score": score,
                    "lead_time": lt
                })
                
        if not candidates:
            e_demand = demands[demand_start_idx]
            net_inv_at_target = inventory
            for ev in all_events:
                if ev['date'] <= e_demand['date']:
                    if ev['type'] == 'supply':
                        net_inv_at_target += ev['qty']
                    elif ev['type'] == 'demand':
                        net_inv_at_target -= ev['qty']
            order_qty = -net_inv_at_target if net_inv_at_target < 0 else e_demand['qty']
            if order_qty <= 0:
                 order_qty = e_demand['qty']
                 
            lt = predict_leadtimes_batch(material_id, material_name, material_spec, [order_qty], lookup_df, models)[0]
            eta = current_date + timedelta(days=int(lt))
            delay_days = max(0, (eta - e_demand['date']).days)
            best_order = {
                "material_id": material_id,
                "material_name": material_name,
                "material_spec": material_spec,
                "order_qty": order_qty,
                "eta": eta,
                "score": delay_days * 100 + order_qty * 0.1,
                "lead_time": lt
            }
        else:
            best_order = min(candidates, key=lambda x: x["score"])
            
        planned_orders.append(best_order)
        
        if len(planned_orders) > len(demands):
            break
            
    return planned_orders

def label_demands(demands, supplies, planned_orders, inventory, planning_horizon_days=90):
    import pandas as pd
    from datetime import datetime, timedelta
    current_date = pd.to_datetime(datetime.now().date())
    horizon_date = current_date + timedelta(days=planning_horizon_days)
    
    events = []
    for s in supplies:
        events.append({'type': 'supply', 'source': 'existing', 'date': s['date'], 'qty': s['qty']})
    for p in planned_orders:
        events.append({'type': 'supply', 'source': 'planned', 'date': p['eta'], 'qty': p['order_qty']})
        
    for d in demands:
        events.append({'type': 'demand', 'date': d['date'], 'qty': d['qty'], 'idx': d['idx']})
        
    events.sort(key=lambda x: (x['date'], 0 if x['type'] == 'supply' else 1))
    
    available_stock = inventory
    available_existing = 0
    available_planned = 0
    
    labels = {}
    
    for e in events:
        if e['type'] == 'supply':
            if e['source'] == 'existing':
                available_existing += e['qty']
            else:
                available_planned += e['qty']
        else:
            req_qty = e['qty']
            
            consume_stock = min(req_qty, available_stock)
            available_stock -= consume_stock
            req_qty -= consume_stock
            
            consume_existing = min(req_qty, available_existing)
            available_existing -= consume_existing
            req_qty -= consume_existing
            
            consume_planned = min(req_qty, available_planned)
            available_planned -= consume_planned
            req_qty -= consume_planned
            
            if req_qty > 0:
                if e['date'] > horizon_date:
                    labels[e['idx']] = 'grey'
                else:
                    labels[e['idx']] = 'red'
            else:
                if consume_planned > 0:
                    labels[e['idx']] = 'orange'
                else:
                    labels[e['idx']] = 'green'
                    
    return labels

def prepare_and_predict():
    import os
    import pandas as pd
    import joblib
    import numpy as np
    from datetime import datetime, timedelta

    # 1. 設定路徑
    BASE_DIR = r'C:\local_file\專題\c1'
    MODEL_DIR = r'C:\local_file\專題\model'
    
    # 輸入檔案
    INPUT_PATH = os.path.join(MODEL_DIR, 'Model缺料物料.csv')
    LOOKUP_PATH = os.path.join(MODEL_DIR, 'C1_Inference_Features_Lookup.csv')
    PURT_PATH = os.path.join(BASE_DIR, 'PURT.csv')
    
    # 載入訓練好的組件
    SCALER_PATH = os.path.join(BASE_DIR, 'x_scaler_num.joblib')
    PT_AMOUNT_PATH = os.path.join(BASE_DIR, 'amount_power_transformer.joblib')
    PT_TARGET_PATH = os.path.join(BASE_DIR, 'target_power_transformer.joblib')
    MODEL_PATH = os.path.join(BASE_DIR, 'XGBoost_Model.joblib')

    if not all([os.path.exists(INPUT_PATH), os.path.exists(LOOKUP_PATH)]):
        print("錯誤：找不到輸入資料表或特徵查找表。")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到模型檔案 {MODEL_PATH}")
        return

    print("正在載入模型組件...")
    models = {
        'scaler_x': joblib.load(SCALER_PATH),
        'pt_amount': joblib.load(PT_AMOUNT_PATH) if os.path.exists(PT_AMOUNT_PATH) else None,
        'pt_target': joblib.load(PT_TARGET_PATH),
        'xgb': joblib.load(MODEL_PATH)
    }

    print("正在載入特徵查找表...")
    lookup_df = pd.read_csv(LOOKUP_PATH)
    lookup_df = lookup_df.drop_duplicates(subset=['品號'], keep='first')

    # =========================================================
    # 任務 1：預測未到貨採購單 (PURT.csv) -> 產出 叫料單預測結果.csv
    # =========================================================
    print("【任務 1】正在預測採購未交單據 (叫料單)...")
    final_purt_df = None
    if os.path.exists(PURT_PATH):
        purt_df = pd.read_csv(PURT_PATH, dtype={'品號': str}, encoding='utf-8-sig')
        purt_df.columns = purt_df.columns.str.strip()
        purt_df['已交數量'] = pd.to_numeric(purt_df['已交數量'], errors='coerce').fillna(0)
        
        prefixes = ('M0', 'M2', 'E', 'K', 'm0', 'm2', 'e', 'k')
        unfulfilled_mask = (purt_df['已交數量'] <= 0) & (purt_df['品號'].str.startswith(prefixes))
        purt_target_df = purt_df[unfulfilled_mask].copy()
        
        if not purt_target_df.empty:
            purt_pred_days = get_predictions(
                df=purt_target_df, lookup_df=lookup_df, 
                id_col='品號', name_col='品名', spec_col='規格', 
                amount_col='採購數量', models=models
            )
            purt_target_df['預測LeadTime'] = purt_pred_days
            purt_target_df['採購日期'] = pd.to_datetime(purt_target_df['採購日期'], errors='coerce')
            default_date = pd.to_datetime(datetime.now().date())
            purt_target_df['採購日期'] = purt_target_df['採購日期'].fillna(default_date)
            
            purt_target_df['預計到料時間'] = purt_target_df.apply(
                lambda row: row['採購日期'] + timedelta(days=int(row['預測LeadTime'])), axis=1
            )
            
            purt_target_df['預計到料時間'] = purt_target_df['預計到料時間'].dt.strftime('%Y-%m-%d')
            purt_target_df['採購日期'] = purt_target_df['採購日期'].dt.strftime('%Y-%m-%d')
            
            order_cols = ['採購單別', '採購單號', '品號', '品名', '規格', '採購數量', '已交數量', '預計到料時間']
            for col in ['採購單別', '採購單號', '採購數量', '已交數量']:
                if col in purt_target_df.columns:
                    purt_target_df[col] = pd.to_numeric(purt_target_df[col], errors='coerce').fillna(0).astype(int)

            available_order_cols = [c for c in order_cols if c in purt_target_df.columns]
            final_purt_df = purt_target_df[available_order_cols]
            
            purt_csv = os.path.join(MODEL_DIR, '叫料單預測結果.csv')
            final_purt_df.to_csv(purt_csv, index=False, encoding='utf-8-sig', sep=',')
            print(f"-> 成功產出 叫料單預測結果.csv：{purt_csv} (共 {len(final_purt_df)} 筆)")
        else:
            print("-> 無符合條件的未交採購單需要預測。")
    else:
        print(f"警告：找不到 PURT 檔案 {PURT_PATH}")

    # =========================================================
    # 任務 2：生成未來叫料單與物料狀態標記
    # =========================================================
    print("【任務 2】正在生成未來叫料單與標記物料狀態...")
    df_shortage = pd.read_csv(INPUT_PATH, encoding='utf-8-sig')
    df_shortage['預計開工日'] = pd.to_datetime(df_shortage['預計開工日'], errors='coerce')
    
    current_date = pd.to_datetime(datetime.now().date())
    
    # 準備現有的 PURT supply events
    purt_supplies = {}
    if final_purt_df is not None and not final_purt_df.empty:
        for _, row in final_purt_df.iterrows():
            mat_id = row['品號']
            if mat_id not in purt_supplies:
                purt_supplies[mat_id] = []
            purt_supplies[mat_id].append({
                'date': pd.to_datetime(row['預計到料時間']),
                'qty': int(row['採購數量'])
            })
            
    all_planned_orders = []
    
    df_shortage['缺料狀態標記'] = 'grey'
    
    for mat_id, group in df_shortage.groupby('材料品號'):
        group = group.sort_values('預計開工日')
        
        material_name = group['材料品名'].iloc[0]
        material_spec = group['材料規格'].iloc[0]
        
        inventory_val = group['庫存數量'].iloc[0]
        try:
            inventory = int(pd.to_numeric(inventory_val))
        except:
            inventory = 0
            
        demands = []
        for orig_idx, row in group.iterrows():
            try:
                qty = int(pd.to_numeric(row['預計用料']))
            except:
                qty = 0
            demands.append({
                'date': row['預計開工日'],
                'qty': qty,
                'idx': orig_idx
            })
            
        supplies = purt_supplies.get(mat_id, [])
        
        planned_orders = find_best_orders_for_material(
            material_id=mat_id,
            material_name=material_name,
            material_spec=material_spec,
            demands=demands,
            supplies=supplies,
            inventory=inventory,
            current_date=current_date,
            lookup_df=lookup_df,
            models=models
        )
        all_planned_orders.extend(planned_orders)
        
        labels = label_demands(demands, supplies, planned_orders, inventory)
        for orig_idx, label in labels.items():
            df_shortage.at[orig_idx, '缺料狀態標記'] = label
            
    tagged_path = os.path.join(MODEL_DIR, 'Model缺料物料_加上預測與標記.csv')
    df_shortage['預計開工日'] = df_shortage['預計開工日'].dt.strftime('%Y-%m-%d')
    df_shortage.to_csv(tagged_path, index=False, encoding='utf-8-sig')
    print(f"-> 成功產出 標記後的缺料表：{tagged_path}")
    
    if all_planned_orders:
        future_orders_df = pd.DataFrame(all_planned_orders)
        future_orders_df['預計到料時間'] = future_orders_df['eta'].dt.strftime('%Y-%m-%d')
        future_orders_df = future_orders_df.rename(columns={
            'material_id': '品號',
            'material_name': '品名',
            'material_spec': '規格',
            'order_qty': '採購數量',
            'lead_time': '預測LeadTime'
        })
        future_orders_df = future_orders_df[['品號', '品名', '規格', '採購數量', '預測LeadTime', '預計到料時間', 'score']]
        
        future_purt_csv = os.path.join(MODEL_DIR, '未來叫料單.csv')
        future_orders_df.to_csv(future_purt_csv, index=False, encoding='utf-8-sig')
        print(f"-> 成功產出 未來叫料單.csv：{future_purt_csv} (共 {len(future_orders_df)} 筆)")
    else:
        print("-> 無需生成未來叫料單。")

    print("-" * 30)
    print("所有預測任務執行完畢！")
    print("-" * 30)

if __name__ == "__main__":
    prepare_and_predict()
