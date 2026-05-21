'''
model預測缺料物料延誤時間，輸出報告：
1. 叫料單預測結果.csv
2. Model缺料物料_加上預測與標記.csv
3. 未來叫料單.csv
'''

import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os
import math
import warnings
from datetime import datetime, timedelta
from sklearn.feature_extraction import FeatureHasher

from material_filters import (
    TARGET_ITEM_PREFIXES,
    filter_to_item_ids,
    filter_to_target_prefixes,
    load_p_item_ids,
)

warnings.filterwarnings('ignore')


# =========================================================
# Utility functions
# =========================================================

def safe_float(value, default=0.0):
    value = pd.to_numeric(value, errors='coerce')
    if pd.isna(value):
        return float(default)
    return float(value)


# =========================================================
# Model prediction functions
# =========================================================

def get_predictions(df, lookup_df, id_col, name_col, spec_col, amount_col, models):
    lookup_cols = [
        '品號', 'Expected_LeadTime_Median',
        'LeadTime_MA3', 'LeadTime_MA5',
        'Hist_Mean_LeadTime', 'Hist_Std_LeadTime', 'Hist_Purchase_Count',
        'Hist_Min_LeadTime', 'Hist_Max_LeadTime', 'Hist_Purchase_Amount',
        'Hist_Actual_Mean', 'Hist_Actual_Std', 'Hist_Actual_Min',
        'Hist_Actual_Max', 'Hist_Actual_CV'
    ]

    lookup_subset = lookup_df[lookup_cols]

    df_merged = pd.merge(
        df,
        lookup_subset,
        left_on=id_col,
        right_on='品號',
        how='left'
    )

    df_merged.fillna(0, inplace=True)

    df_merged[amount_col] = pd.to_numeric(
        df_merged[amount_col],
        errors='coerce'
    ).fillna(0)

    if models['pt_amount']:
        df_merged['Amount_PT'] = models['pt_amount'].transform(
            df_merged[[amount_col]].fillna(0)
        ).flatten()
    else:
        df_merged['Amount_PT'] = np.log1p(df_merged[amount_col].fillna(0))

    df_merged['Expected_LeadTime_Log'] = np.log1p(
        df_merged['Expected_LeadTime_Median'].fillna(0).clip(lower=0)
    )

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

    sparse_blocks = []

    id_str_upper = df_merged[id_col].astype(str).str.upper()
    cat_mat = np.column_stack([
        id_str_upper.str.startswith('M0').astype(np.int8),
        id_str_upper.str.startswith('M2').astype(np.int8),
        id_str_upper.str.startswith('K').astype(np.int8),
        id_str_upper.str.startswith('E').astype(np.int8)
    ])
    sparse_blocks.append(sp.csr_matrix(cat_mat))

    hasher_id_full = FeatureHasher(n_features=256, input_type='string')
    hash_full_sp = hasher_id_full.transform(
        df_merged[id_col].astype(str).apply(lambda x: [x])
    )
    sparse_blocks.append(hash_full_sp)

    hasher_id_split = FeatureHasher(n_features=128, input_type='string')
    id_str_pad = df_merged[id_col].astype(str).apply(lambda x: x.ljust(14, ' '))

    hash_split_sp = sp.hstack([
        hasher_id_split.transform(id_str_pad.str[0:2].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[2:6].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[6:10].apply(lambda x: [x])),
        hasher_id_split.transform(id_str_pad.str[10:14].apply(lambda x: [x]))
    ], format='csr')
    sparse_blocks.append(hash_split_sp)

    df_merged['Name_Spec'] = (
        df_merged[name_col].astype(str)
        + "_"
        + df_merged[spec_col].astype(str).fillna('')
    )

    hasher_spec = FeatureHasher(n_features=512, input_type='string')
    hash_spec_sp = hasher_spec.transform(
        df_merged['Name_Spec'].apply(lambda x: [x])
    )
    sparse_blocks.append(hash_spec_sp)

    X_final = sp.hstack(
        [sp.csr_matrix(X_num_scaled)] + sparse_blocks,
        format='csr'
    )

    y_pred_pt = models['xgb'].predict(X_final)
    y_pred_days = models['pt_target'].inverse_transform(
        y_pred_pt.reshape(-1, 1)
    ).flatten()

    return np.maximum(np.round(y_pred_days), 1).astype(int)


def predict_leadtimes_batch(
    material_id,
    material_name,
    material_spec,
    order_qtys,
    lookup_df,
    models
):
    if not order_qtys:
        return []

    df_temp = pd.DataFrame({
        '品號': [material_id] * len(order_qtys),
        '品名': [material_name] * len(order_qtys),
        '規格': [material_spec] * len(order_qtys),
        '採購數量': order_qtys
    })

    preds = get_predictions(
        df_temp,
        lookup_df,
        '品號',
        '品名',
        '規格',
        '採購數量',
        models
    )

    return preds


# =========================================================
# Core order planning function
# =========================================================

def find_best_orders_for_material(
    material_id,
    material_name,
    material_spec,
    demands,
    supplies,
    inventory,
    current_date,
    lookup_df,
    models,
    x_days=30,
    planning_horizon_days=90
):
    """
    單一物料、本次 review cycle 的新叫料單決策。

    顏色定義：
    - yellow：本次新訂單可以準時抵達，會進本次叫料單。
    - red：本次應該處理，但無法準時抵達，仍會進本次叫料單。
    - grey：本次不處理，交給之後叫料單處理，不進本次叫料單。

    關鍵規則：
    - demands 傳進來時，qty 必須是「缺料增量」，不是累積缺料數量。
    - 本函數不再用庫存流重新找 shortage_idx。
    - 本輪 x_days 內所有缺料需求都先進 candidate_pool。
    - 依時間順序：
        1. 不能準時者先標 red，且放進本次叫料單。
        2. 第一個能準時者為 anchor，標 yellow。
        3. 後續需求若加入後不破壞 anchor 準時性，標 yellow。
        4. 否則標 grey，留給之後叫料單。
    """

    VALID_COLORS = {"yellow", "grey", "red"}

    current_date = pd.to_datetime(current_date).normalize()
    planned_orders = []

    horizon_date = current_date + timedelta(days=planning_horizon_days)
    review_end = current_date + timedelta(days=x_days)

    demands = [
        {
            **d,
            "date": pd.to_datetime(d["date"]).normalize(),
            "qty": safe_float(d.get("qty", 0)),
        }
        for d in demands
        if pd.notna(d.get("date")) and safe_float(d.get("qty", 0)) > 0
    ]

    supplies = [
        {
            **s,
            "date": pd.to_datetime(s["date"]).normalize(),
            "qty": safe_float(s.get("qty", 0)),
        }
        for s in supplies
        if pd.notna(s.get("date"))
    ]

    demands = sorted(demands, key=lambda d: d["date"])
    supplies = sorted(supplies, key=lambda s: s["date"])

    if not demands:
        return planned_orders

    def qty_of(events):
        return sum(safe_float(e.get("qty", 0)) for e in events)

    def predict_order(order_qty):
        if order_qty <= 0:
            return 0, current_date

        lt = predict_leadtimes_batch(
            material_id,
            material_name,
            material_spec,
            [order_qty],
            lookup_df,
            models
        )[0]

        eta = current_date + timedelta(days=math.ceil(float(lt)))
        return lt, eta

    # =========================================================
    # Step 1: 本輪 review window 內所有缺料需求都進處理池
    # =========================================================
    candidate_pool = [
        d for d in demands
        if d["date"] <= review_end and d["date"] <= horizon_date
    ]

    if not candidate_pool:
        return planned_orders

    demand_colors = {}

    for d in demands:
        if d["date"] <= horizon_date:
            demand_colors[d["idx"]] = "grey"

    candidates = []

    red_events = []
    yellow_events = []
    covered_events = []

    anchor_event = None
    anchor_index = None
    anchor_date = None

    selected_order_qty = 0.0
    selected_lead_time = None
    selected_eta = None
    selected_target_date = None

    # =========================================================
    # Step 2: 依時間順序，先把不可能準時的需求納入本次處理池
    # =========================================================
    for k, d in enumerate(candidate_pool):
        test_events = red_events + [d]
        test_order_qty = qty_of(test_events)

        lt, eta = predict_order(test_order_qty)

        if eta <= d["date"]:
            anchor_event = d
            anchor_index = k
            anchor_date = d["date"]

            demand_colors[d["idx"]] = "yellow"
            yellow_events.append(d)

            selected_order_qty = test_order_qty
            selected_lead_time = lt
            selected_eta = eta
            selected_target_date = d["date"]

            candidates.append({
                "process_id": d["idx"],
                "target_date": d["date"],
                "order_qty": test_order_qty,
                "eta": eta,
                "lead_time": lt,
                "decision": "yellow",
                "reason": "First achievable anchor demand after including prior red demands."
            })

            break

        else:
            demand_colors[d["idx"]] = "red"
            red_events.append(d)

            candidates.append({
                "process_id": d["idx"],
                "target_date": d["date"],
                "order_qty": test_order_qty,
                "eta": eta,
                "lead_time": lt,
                "decision": "red",
                "reason": "This demand should be handled now, but cannot arrive on time."
            })

    # =========================================================
    # Step 3: 沒有 anchor，本輪全部 red，但仍要叫料
    # =========================================================
    if anchor_event is None:
        for d in candidate_pool:
            demand_colors[d["idx"]] = "red"

        red_events = candidate_pool
        covered_events = red_events

        order_qty = qty_of(covered_events)
        lt, eta = predict_order(order_qty)

        red_processes = [d["idx"] for d in red_events]
        red_qty = qty_of(red_events)

        best_order = {
            "material_id": material_id,
            "material_name": material_name,
            "material_spec": material_spec,

            "order_color": "red",
            "status": "red",

            "order_qty": order_qty,
            "eta": eta,
            "lead_time": lt,
            "score": red_qty * 1000 + order_qty * 0.1,

            "target_date": candidate_pool[-1]["date"],
            "earliest_required": candidate_pool[0]["date"],
            "covered_until": candidate_pool[-1]["date"],

            "anchor_process": None,
            "anchor_date": None,

            "covered_processes": red_processes,
            "yellow_processes": [],
            "red_processes": red_processes,
            "grey_processes": [
                idx for idx, color in demand_colors.items()
                if color == "grey"
            ],

            "covered_qty_within_x": red_qty,
            "yellow_qty": 0,
            "red_qty": red_qty,
            "grey_qty": 0,
            "uncovered_qty_within_x": 0,

            "on_time_processes": [],
            "on_time_count": 0,
            "on_time_qty": 0,

            "late_qty": red_qty,

            "forced_merge": True,
            "forced_late_qty": red_qty,
            "forced_merge_additional_order_qty": red_qty,

            "stopped_by_grey": False,

            "demand_colors": demand_colors,
            "candidate_trace": candidates,

            "reason": (
                "No demand in this review cycle can arrive on time. "
                "All current-cycle demands are marked red and still included in this order."
            )
        }

        assert set(best_order["demand_colors"].values()).issubset(VALID_COLORS)

        planned_orders.append(best_order)
        return planned_orders

    # =========================================================
    # Step 4: 有 anchor，先把 red prefix + anchor 放進本次叫料單
    # =========================================================
    covered_events = red_events + yellow_events
    stopped_by_grey = False

    # =========================================================
    # Step 5: 往後加入需求
    # =========================================================
    for k in range(anchor_index + 1, len(candidate_pool)):
        d = candidate_pool[k]

        test_events = covered_events + [d]
        test_order_qty = qty_of(test_events)

        test_lt, test_eta = predict_order(test_order_qty)

        if test_eta <= anchor_date:
            demand_colors[d["idx"]] = "yellow"

            yellow_events.append(d)
            covered_events.append(d)

            selected_order_qty = test_order_qty
            selected_lead_time = test_lt
            selected_eta = test_eta
            selected_target_date = d["date"]

            candidates.append({
                "process_id": d["idx"],
                "target_date": d["date"],
                "order_qty": test_order_qty,
                "eta": test_eta,
                "lead_time": test_lt,
                "decision": "yellow",
                "reason": "Added to current order without breaking anchor on-time requirement."
            })

        else:
            demand_colors[d["idx"]] = "grey"
            stopped_by_grey = True

            candidates.append({
                "process_id": d["idx"],
                "target_date": d["date"],
                "order_qty": test_order_qty,
                "eta": test_eta,
                "lead_time": test_lt,
                "decision": "grey",
                "reason": "Left for future order because adding it would make anchor late."
            })

            for j in range(k + 1, len(candidate_pool)):
                later_d = candidate_pool[j]
                demand_colors[later_d["idx"]] = "grey"

            break

    # =========================================================
    # Step 6: 彙總結果
    # =========================================================
    yellow_events = [
        d for d in candidate_pool
        if demand_colors.get(d["idx"]) == "yellow"
    ]

    red_events = [
        d for d in candidate_pool
        if demand_colors.get(d["idx"]) == "red"
    ]

    grey_events = [
        d for d in candidate_pool
        if demand_colors.get(d["idx"]) == "grey"
    ]

    yellow_processes = [d["idx"] for d in yellow_events]
    red_processes = [d["idx"] for d in red_events]

    grey_processes = [
        idx for idx, color in demand_colors.items()
        if color == "grey"
    ]

    covered_processes = [d["idx"] for d in covered_events]

    yellow_qty = qty_of(yellow_events)
    red_qty = qty_of(red_events)
    grey_qty = qty_of(grey_events)

    covered_qty_within_x = yellow_qty + red_qty
    uncovered_qty_within_x = grey_qty

    forced_merge = red_qty > 0

    total_delay_days = sum(
        max(0, (selected_eta - d["date"]).days)
        for d in covered_events
    )

    score = (
        total_delay_days * 1000
        + uncovered_qty_within_x * 100
        + selected_order_qty * 0.1
    )

    best_order = {
        "material_id": material_id,
        "material_name": material_name,
        "material_spec": material_spec,

        "order_color": "yellow",
        "status": "yellow",

        "order_qty": selected_order_qty,
        "eta": selected_eta,
        "lead_time": selected_lead_time,
        "score": score,

        "target_date": selected_target_date,
        "earliest_required": anchor_event["date"],
        "covered_until": selected_target_date,

        "anchor_process": anchor_event["idx"],
        "anchor_date": anchor_date,

        "covered_processes": covered_processes,
        "yellow_processes": yellow_processes,
        "red_processes": red_processes,
        "grey_processes": grey_processes,

        "covered_qty_within_x": covered_qty_within_x,
        "yellow_qty": yellow_qty,
        "red_qty": red_qty,
        "grey_qty": grey_qty,
        "uncovered_qty_within_x": uncovered_qty_within_x,

        "on_time_processes": yellow_processes,
        "on_time_count": len(yellow_processes),
        "on_time_qty": yellow_qty,

        "late_qty": red_qty,

        "forced_merge": forced_merge,
        "forced_late_qty": red_qty,
        "forced_merge_additional_order_qty": red_qty,

        "stopped_by_grey": stopped_by_grey,

        "demand_colors": demand_colors,
        "candidate_trace": candidates,

        "reason": (
            "Red demands are handled first because they should have been processed already. "
            "Yellow demands are handled by the new order and can arrive on time. "
            "Grey demands are left for future orders."
        )
    }

    assert set(best_order["demand_colors"].values()).issubset(VALID_COLORS)

    planned_orders.append(best_order)
    return planned_orders


# =========================================================
# Label function
# =========================================================

def label_demands(
    demands,
    supplies,
    planned_orders,
    inventory,
    current_date,
    planning_horizon_days=90
):
    """
    回傳：labels, etas, candidate_etas

    labels：每個需求 idx 的顏色。
    etas：最終叫料單 ETA。
    candidate_etas：每個 candidate 自己測試時的 ETA。
    """

    VALID_COLORS = {"yellow", "grey", "red"}

    current_date = pd.to_datetime(current_date).normalize()
    horizon_date = current_date + timedelta(days=planning_horizon_days)

    labels = {}
    etas = {}
    candidate_etas = {}

    for d in demands:
        if pd.isna(d.get("date")):
            continue

        d_date = pd.to_datetime(d["date"]).normalize()

        if d_date <= horizon_date:
            labels[d["idx"]] = "grey"

    for p in planned_orders:
        demand_colors = p.get("demand_colors", {})
        order_eta = p.get("eta", None)

        if order_eta is not None:
            order_eta = pd.to_datetime(order_eta).normalize()

        # 1. 先把每個 candidate 自己的 ETA 寫入 candidate_etas
        for trace in p.get("candidate_trace", []):
            idx = trace.get("process_id")
            trace_eta = trace.get("eta", None)

            if idx is not None and trace_eta is not None:
                candidate_etas[idx] = pd.to_datetime(trace_eta).normalize()

        # 2. 再貼正式顏色與最終叫料單 ETA
        for idx, color in demand_colors.items():
            color = str(color).lower()

            if color not in VALID_COLORS:
                raise ValueError(
                    f"Invalid color '{color}' for demand idx {idx}. "
                    f"Allowed colors: {VALID_COLORS}"
                )

            current_color = labels.get(idx, "grey")

            if color == "red":
                labels[idx] = "red"

                if order_eta is not None:
                    etas[idx] = order_eta

            elif color == "yellow":
                if current_color != "red":
                    labels[idx] = "yellow"

                    if order_eta is not None:
                        etas[idx] = order_eta

            elif color == "grey":
                labels.setdefault(idx, "grey")

    for idx, color in labels.items():
        if color not in VALID_COLORS:
            raise ValueError(
                f"Invalid label '{color}' at demand idx {idx}. "
                f"Allowed colors: {VALID_COLORS}"
            )

    return labels, etas, candidate_etas

# =========================================================
# Main pipeline
# =========================================================

def prepare_and_predict():
    BASE_DIR = r'C:\local_file\專題\c1'
    MODEL_DIR = r'C:\local_file\專題\model'
    ERP_PATH = r'C:\local_file\專題\ERP_Table.xlsx'

    INPUT_PATH = os.path.join(MODEL_DIR, 'Model缺料物料.csv')
    LOOKUP_PATH = os.path.join(MODEL_DIR, 'C1_Inference_Features_Lookup.csv')
    PURT_PATH = os.path.join(BASE_DIR, 'PURT.csv')

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

    try:
        p_item_ids = load_p_item_ids(ERP_PATH)
    except (FileNotFoundError, ValueError) as e:
        print(f"錯誤：無法套用 INVMB 品號屬性 P 過濾，停止預測。原因：{e}")
        return

    print(f"已載入 INVMB 品號屬性 = 'P' 的品號數：{len(p_item_ids)}")

    print("正在載入模型組件...")

    models = {
        'scaler_x': joblib.load(SCALER_PATH),
        'pt_amount': joblib.load(PT_AMOUNT_PATH) if os.path.exists(PT_AMOUNT_PATH) else None,
        'pt_target': joblib.load(PT_TARGET_PATH),
        'xgb': joblib.load(MODEL_PATH)
    }

    print("正在載入特徵查找表...")

    lookup_df = pd.read_csv(LOOKUP_PATH)
    lookup_before_prefix_count = len(lookup_df)
    lookup_df = filter_to_target_prefixes(lookup_df, '品號')
    lookup_before_p_count = len(lookup_df)
    lookup_df = filter_to_item_ids(lookup_df, '品號', p_item_ids)
    lookup_df = lookup_df.drop_duplicates(subset=['品號'], keep='first')
    print(
        "-> 特徵查找表前綴 + P 過濾："
        f"{lookup_before_prefix_count} -> {lookup_before_p_count} -> {len(lookup_df)} 筆"
    )

    # =========================================================
    # 任務 1：預測未到貨採購單 PURT.csv
    # =========================================================

    print("【任務 1】正在預測採購未交單據 (叫料單)...")

    final_purt_df = None

    if os.path.exists(PURT_PATH):
        purt_df = pd.read_csv(
            PURT_PATH,
            dtype={'品號': str},
            encoding='utf-8-sig'
        )

        purt_df.columns = purt_df.columns.str.strip()
        purt_df['品號'] = purt_df['品號'].astype(str).str.strip()
        purt_df['已交數量'] = pd.to_numeric(
            purt_df['已交數量'],
            errors='coerce'
        ).fillna(0)

        prefix_mask = purt_df['品號'].str.startswith(TARGET_ITEM_PREFIXES)
        p_item_mask = purt_df['品號'].isin(p_item_ids)
        unfulfilled_mask = (
            (purt_df['已交數量'] <= 0)
            & prefix_mask
            & p_item_mask
        )

        purt_target_df = purt_df[unfulfilled_mask].copy()
        print(f"-> PURT 未交採購單前綴 + P 過濾後需預測筆數：{len(purt_target_df)}")

        if not purt_target_df.empty:
            purt_pred_days = get_predictions(
                df=purt_target_df,
                lookup_df=lookup_df,
                id_col='品號',
                name_col='品名',
                spec_col='規格',
                amount_col='採購數量',
                models=models
            )

            purt_target_df['預測LeadTime'] = purt_pred_days
            purt_target_df['採購日期'] = pd.to_datetime(
                purt_target_df['採購日期'],
                errors='coerce'
            )

            purt_target_df['採購日期'] = purt_target_df['採購日期'].fillna(
                pd.to_datetime(datetime.now().date())
            )

            purt_target_df['預計到料時間'] = purt_target_df.apply(
                lambda row: row['採購日期'] + timedelta(days=int(row['預測LeadTime'])),
                axis=1
            )

            purt_target_df['預計到料時間'] = purt_target_df['預計到料時間'].dt.strftime('%Y-%m-%d')
            purt_target_df['採購日期'] = purt_target_df['採購日期'].dt.strftime('%Y-%m-%d')

            order_cols = [
                '採購單別',
                '採購單號',
                '品號',
                '品名',
                '規格',
                '採購數量',
                '已交數量',
                '預計到料時間'
            ]

            for col in ['採購單別', '採購單號', '採購數量', '已交數量']:
                if col in purt_target_df.columns:
                    purt_target_df[col] = pd.to_numeric(
                        purt_target_df[col],
                        errors='coerce'
                    ).fillna(0).astype(int)

            available_order_cols = [
                c for c in order_cols
                if c in purt_target_df.columns
            ]

            final_purt_df = purt_target_df[available_order_cols]

            purt_csv = os.path.join(MODEL_DIR, '叫料單預測結果.csv')
            final_purt_df.to_csv(
                purt_csv,
                index=False,
                encoding='utf-8-sig',
                sep=','
            )

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
    df_shortage.columns = df_shortage.columns.str.strip()
    shortage_before_prefix_count = len(df_shortage)
    df_shortage = filter_to_target_prefixes(df_shortage, '材料品號')
    shortage_before_p_count = len(df_shortage)
    df_shortage = filter_to_item_ids(df_shortage, '材料品號', p_item_ids)
    print(
        "-> 缺料物料表前綴 + P 過濾："
        f"{shortage_before_prefix_count} -> {shortage_before_p_count} -> {len(df_shortage)} 筆"
    )

    df_shortage['預計開工日'] = pd.to_datetime(
        df_shortage['預計開工日'],
        errors='coerce'
    )

    current_date = pd.to_datetime(datetime.now().date())

    purt_supplies = {}

    if final_purt_df is not None and not final_purt_df.empty:
        for _, row in final_purt_df.iterrows():
            mat_id = row['品號']

            if mat_id not in purt_supplies:
                purt_supplies[mat_id] = []

            purt_supplies[mat_id].append({
                'date': pd.to_datetime(row['預計到料時間']),
                'qty': safe_float(row['採購數量'])
            })

    all_planned_orders = []

    df_shortage['缺料狀態標記'] = 'grey'
    df_shortage['預計到料時間'] = ''
    df_shortage['候選預測到料時間'] = ''

    use_shortage_qty = '缺料數量' in df_shortage.columns

    for mat_id, group in df_shortage.groupby('材料品號'):
        group = group.sort_values('預計開工日', kind='mergesort').copy()

        material_name = group['材料品名'].iloc[0]
        material_spec = group['材料規格'].iloc[0]

        inventory_val = group['庫存數量'].iloc[0] if '庫存數量' in group.columns else 0
        raw_inventory = safe_float(inventory_val)

        demands = []

        if use_shortage_qty:
            # 缺料數量是累積缺口，不是每列需求量。
            # 所以必須轉成每列新增缺口，避免 20+40+60... 加成超大數。
            inventory = 0.0

            group['_缺料數量_num'] = pd.to_numeric(
                group['缺料數量'],
                errors='coerce'
            ).fillna(0)

            group['_缺料增量'] = group['_缺料數量_num'].diff().fillna(
                group['_缺料數量_num']
            )

            group['_缺料增量'] = group['_缺料增量'].clip(lower=0)

            for orig_idx, row in group.iterrows():
                if pd.isna(row['預計開工日']):
                    continue

                qty = safe_float(row['_缺料增量'])

                if qty <= 0:
                    continue

                demands.append({
                    'date': row['預計開工日'],
                    'qty': qty,
                    'idx': orig_idx
                })

        else:
            inventory = raw_inventory

            for orig_idx, row in group.iterrows():
                if pd.isna(row['預計開工日']):
                    continue

                qty = safe_float(row.get('預計用料', 0))

                if qty <= 0:
                    continue

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
            models=models,
            x_days=30,
            planning_horizon_days=90
        )

        all_planned_orders.extend(planned_orders)

        labels, etas, candidate_etas = label_demands(
            demands=demands,
            supplies=supplies,
            planned_orders=planned_orders,
            inventory=inventory,
            current_date=current_date,
            planning_horizon_days=90
        )

        for orig_idx, label in labels.items():
            df_shortage.at[orig_idx, '缺料狀態標記'] = label

        # 最終叫料單 ETA：同一張單會相同
            eta = etas.get(orig_idx)

            if eta is not None and pd.notnull(eta):
                df_shortage.at[orig_idx, '預計到料時間'] = pd.to_datetime(eta).strftime('%Y-%m-%d')

            # 每個 candidate 自己測試時的 ETA：用來驗證每個數量是否有丟進模型
            candidate_eta = candidate_etas.get(orig_idx)

            if candidate_eta is not None and pd.notnull(candidate_eta):
                df_shortage.at[orig_idx, '候選預測到料時間'] = pd.to_datetime(candidate_eta).strftime('%Y-%m-%d')



    tagged_path = os.path.join(MODEL_DIR, 'Model缺料物料_加上預測與標記.csv')

    df_shortage['預計開工日'] = df_shortage['預計開工日'].dt.strftime('%Y-%m-%d')
    df_shortage.to_csv(tagged_path, index=False, encoding='utf-8-sig')

    print(f"-> 成功產出 標記後的缺料表：{tagged_path}")

    if all_planned_orders:
        future_orders_df = pd.DataFrame(all_planned_orders)

        future_orders_df['預計到料時間'] = pd.to_datetime(
            future_orders_df['eta']
        ).dt.strftime('%Y-%m-%d')

        future_orders_df = future_orders_df.rename(columns={
            'material_id': '品號',
            'material_name': '品名',
            'material_spec': '規格',
            'order_qty': '採購數量',
            'lead_time': '預測LeadTime'
        })

        export_cols = [
            '品號',
            '品名',
            '規格',
            '採購數量',
            '預測LeadTime',
            '預計到料時間',
            'score',
            'status'
        ]

        export_cols = [
            c for c in export_cols
            if c in future_orders_df.columns
        ]

        future_orders_df = future_orders_df[export_cols]

        future_orders_df = future_orders_df.sort_values(
            '預計到料時間'
        ).drop_duplicates(
            subset=['品號'],
            keep='first'
        )

        future_purt_csv = os.path.join(MODEL_DIR, '未來叫料單.csv')

        future_orders_df.to_csv(
            future_purt_csv,
            index=False,
            encoding='utf-8-sig'
        )

        print(f"-> 成功產出 未來叫料單.csv：{future_purt_csv} (共 {len(future_orders_df)} 筆)")

    else:
        print("-> 無需生成未來叫料單。")

    print("-" * 30)
    print("所有預測任務執行完畢！")
    print("-" * 30)


if __name__ == "__main__":
    prepare_and_predict()
