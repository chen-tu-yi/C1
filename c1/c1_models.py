import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import joblib
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.base import clone
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings('ignore', category=UserWarning, message='.*valid feature names.*')

png_dir = r"C:\local_file\專題\png\c1"
os.makedirs(png_dir, exist_ok=True)

# Tree Models
# from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# from catboost import CatBoostRegressor

# NN Models
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks


# =========================================================
# Error Map Function
# =========================================================


def generate_error_map(y_true_days, y_pred_days, model_name, output_dir):
    """
    Error Map:
    用 validation set 的 actual lead time 與 predicted lead time 建立誤差分析圖。

    最終保留三個指標：
    1. MAE：平均誤差天數
    2. MAPE：平均誤差百分比
    3. 低估時平均延後天數：只看 actual > predicted 的情況，平均實際晚幾天
    """

    model_name_safe = str(model_name).replace(" ", "_").replace("/", "_")

    df_error = pd.DataFrame({
        "Actual_Days": y_true_days,
        "Predicted_Days": y_pred_days
    })

    # error > 0 代表模型低估：實際到料天數比預測更久
    df_error["Error"] = df_error["Actual_Days"] - df_error["Predicted_Days"]

    # MAE 用
    df_error["Abs_Error"] = df_error["Error"].abs()

    # MAPE 用，避免 actual_days = 0 導致除以 0
    df_error["APE"] = (
        df_error["Abs_Error"]
        / np.maximum(np.abs(df_error["Actual_Days"]), 1)
    ) * 100

    # 低估時平均延後天數用
    df_error["Underestimate_Days"] = np.where(
        df_error["Error"] > 0,
        df_error["Error"],
        np.nan
    )

    # 依模型預測到料天數分區
    bins = [-np.inf, 3, 10, np.inf]
    labels = ["3天內", "4–10天", "超過10天"]

    df_error["模型預測到料天數"] = pd.cut(
        df_error["Predicted_Days"],
        bins=bins,
        labels=labels
    )

    error_map = (
        df_error
        .groupby("模型預測到料天數", observed=False)
        .agg(
            資料筆數=("Error", "count"),
            MAE=("Abs_Error", "mean"),
            MAPE=("APE", "mean"),
            低估時平均延後天數=("Underestimate_Days", "mean")
        )
        .reset_index()
    )

    # 儲存明細與彙總
    detail_path = os.path.join(output_dir, f"C1_{model_name_safe}_Error_Detail.csv")
    map_path = os.path.join(output_dir, f"C1_{model_name_safe}_Error_Map.csv")

    df_error.to_csv(detail_path, index=False, encoding="utf-8-sig")
    error_map.to_csv(map_path, index=False, encoding="utf-8-sig")

    # 畫圖
    heatmap_cols = [
        "MAE",
        "MAPE",
        "低估時平均延後天數"
    ]

    heatmap_data = error_map[heatmap_cols].values

    plt.figure(figsize=(9, 4))
    plt.imshow(heatmap_data, aspect="auto", cmap="Reds")
    plt.colorbar(label="數值")

    plt.xticks(
        ticks=np.arange(len(heatmap_cols)),
        labels=["MAE\n平均誤差天數", "MAPE\n平均誤差百分比", "低估時\n平均延後天數"],
        rotation=0
    )

    plt.yticks(
        ticks=np.arange(len(error_map["模型預測到料天數"])),
        labels=error_map["模型預測到料天數"].astype(str)
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]

            if pd.notna(value):
                if heatmap_cols[j] == "MAPE":
                    text = f"{value:.1f}%"
                else:
                    text = f"{value:.1f}天"

                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center"
                )

    plt.title(f"{model_name} 到料天數預測誤差分析")
    plt.xlabel("誤差指標")
    plt.ylabel("模型預測到料天數")
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"C1_{model_name_safe}_Error_Map.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"{model_name} error map saved:")
    print(f"- {map_path}")
    print(f"- {png_path}")

    return error_map


# =========================================================
# 1. 讀取與資料清洗
# =========================================================

print("讀取 Sparse Matrix 特徵資料...")
X_train = sp.load_npz('C1_ML_Training_X.npz')
y_train = np.load('C1_ML_Training_y.npy')
X_val = sp.load_npz('C1_ML_Test_X.npz')
y_val = np.load('C1_ML_Test_y.npy')

# Sparse Matrix 不包含欄位名稱，在此建立通用特徵名稱
feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
total_samples = X_train.shape[0] + X_val.shape[0]

print("讀取目標變數轉換器 (Inverse Transform 用)...")
pt_y = joblib.load('target_power_transformer.joblib')

# 讀取 val set 承諾交期天數（預計進貨天數），供 Late Recall 計算
try:
    required_days_val = np.load('C1_ML_Test_required_days.npy')
    print(f"已載入 val set 承諾交期天數 ({len(required_days_val)} 筆)")
except FileNotFoundError:
    required_days_val = None
    print("警告：C1_ML_Test_required_days.npy 不存在，Late Recall 將跳過計算（請重新執行 c1_pre.py）")


# =========================================================
# 2. 定義 Tree 型模型
# =========================================================

models_dict = {
    # "LightGBM": LGBMRegressor(
    #     n_estimators=1000,
    #     learning_rate=0.05,
    #     random_state=42,
    #     n_jobs=-1
    # ),
    # "CatBoost": CatBoostRegressor(
    #     n_estimators=1000,
    #     learning_rate=0.05,
    #     depth=6,
    #     random_seed=42,
    #     verbose=0
    # ),
    "XGBoost": XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse'
    )
}


# =========================================================
# 3. 定義神經網路結構 (MLP)
# =========================================================

# def build_nn_model(input_shape):
#     model = models.Sequential([
#         layers.Dense(64, activation='relu', input_shape=(input_shape,)),
#         layers.Dropout(0.2),

#         layers.Dense(32, activation='relu'),
#         layers.Dropout(0.1),

#         layers.Dense(16, activation='relu'),

#         layers.Dense(1)
#     ])

#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model


# =========================================================
# 4. 執行訓練與 Log 紀錄
# =========================================================

results = []
importance_list = []

with open("training_log.txt", "w", encoding="utf-8") as f_log:
    f_log.write("=== C1 採購延遲預測 訓練實驗紀錄 ===\n")
    f_log.write(f"執行時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_log.write(f"樣本數: {total_samples}, 特徵數: {X_train.shape[1]}\n\n")

    # =========================================================
    # (A) 訓練 Tree 型模型
    # =========================================================

    for name, model in models_dict.items():
        print(f"\n正在訓練 {name}...")
        start = time.time()

        if name == "CatBoost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)]
            )

        # 儲存 XGBoost 模型供推論使用
        if name == "XGBoost":
            joblib.dump(model, 'XGBoost_Model.joblib')
            print("XGBoost 模型已儲存為 XGBoost_Model.joblib")

        duration = time.time() - start
        y_pred = model.predict(X_val)

        # PT 空間指標
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 真實天數空間
        y_val_inv = pt_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_inv = pt_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        rmse_inv = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))
        mae_inv = mean_absolute_error(y_val_inv, y_pred_inv)
        r2_inv = r2_score(y_val_inv, y_pred_inv)

        # Adjusted MAPE
        mape_inv = np.mean(
            np.abs(y_val_inv - y_pred_inv) / np.maximum(np.abs(y_val_inv), 1)
        ) * 100

        # 生成 Error Map
        if name == "XGBoost":
            error_map_df = generate_error_map(
                y_true_days=y_val_inv,
                y_pred_days=y_pred_inv,
                model_name=name,
                output_dir=png_dir
            )

        # =========================================================
        # Recall 計算
        # Late Recall       = TP_late / (TP_late + FN_late)
        #                     只算「遲到」：actual > req
        # Non-ontime Recall = TP_nonot / (TP_nonot + FN_nonot)
        #                     「非準時」：遲到 OR 提早 (actual != req)
        # Ontime Recall     = TN / (TN + FP)
        #                     「準時」：actual == req
        # =========================================================
        late_recall = None
        nonot_recall = None
        ontime_recall = None
        late_recall_str = "N/A (缺少承諾交期資料)"
        nonot_recall_str = "N/A (缺少承諾交期資料)"
        ontime_recall_str = "N/A (缺少承諾交期資料)"

        if required_days_val is not None:
            req = required_days_val.astype(float)

            # 遲到：actual > req
            actual_late = y_val_inv > req
            predicted_late = y_pred_inv > req

            # 提早：actual < req
            actual_early = y_val_inv < req
            predicted_early = y_pred_inv < req

            # 非準時：遲到 OR 提早 (actual != req)
            actual_nonot = actual_late | actual_early
            predicted_nonot = predicted_late | predicted_early

            # 準時：actual == req
            actual_ontime = ~actual_nonot
            predicted_ontime = ~predicted_nonot

            # Late Recall = 遲到訂單中，模型預測遲到的比例
            n_actual_late = actual_late.sum()
            if n_actual_late > 0:
                tp_late = (actual_late & predicted_late).sum()
                late_recall = tp_late / n_actual_late
                late_recall_str = (
                    f"{late_recall:.4f} "
                    f"(TP={tp_late}, TP+FN={n_actual_late})"
                )
            else:
                late_recall_str = "N/A (val set 中無實際延誤訂單)"

            # Non-ontime Recall = 非準時訂單(提早+遲到)中，模型預測非準時的比例
            n_actual_nonot = actual_nonot.sum()
            if n_actual_nonot > 0:
                tp_nonot = (actual_nonot & predicted_nonot).sum()
                nonot_recall = tp_nonot / n_actual_nonot
                nonot_recall_str = (
                    f"{nonot_recall:.4f} "
                    f"(TP={tp_nonot}, TP+FN={n_actual_nonot})"
                )
            else:
                nonot_recall_str = "N/A (val set 中無非準時訂單)"

            # Ontime Recall = 準時訂單中，模型預測準時的比例
            n_actual_ontime = actual_ontime.sum()
            if n_actual_ontime > 0:
                tn = (actual_ontime & predicted_ontime).sum()
                ontime_recall = tn / n_actual_ontime
                ontime_recall_str = (
                    f"{ontime_recall:.4f} "
                    f"(TN={tn}, TN+FP={n_actual_ontime})"
                )
            else:
                ontime_recall_str = "N/A (val set 中無實際準時訂單)"

        res = (
            f"[{name}] "
            f"PT Space - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f} | "
            f"Real Space - RMSE: {rmse_inv:.4f}, MAE: {mae_inv:.4f}, "
            f"R2: {r2_inv:.4f}, MAPE: {mape_inv:.2f}% | "
            f"Late Recall: {late_recall_str} | "
            f"Non-ontime Recall: {nonot_recall_str} | "
            f"Ontime Recall: {ontime_recall_str} | "
            f"Time: {duration:.2f}s\n"
        )

        print(res.strip())
        f_log.write(res)

        results.append({
            "Model": name,
            "RMSE (PT)": rmse,
            "MAE (PT)": mae,
            "R2 (PT)": r2,
            "RMSE (Days)": rmse_inv,
            "MAE (Days)": mae_inv,
            "R2 (Days)": r2_inv,
            "MAPE(%)": mape_inv,
            "Late Recall": late_recall,
            "Non-ontime Recall": nonot_recall,
            "Ontime Recall": ontime_recall
        })

        # Feature Importance
        feat_imp = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        importance_list.append(feat_imp.to_frame(name=name))

        # Learning Curve
        try:
            plt.figure(figsize=(10, 6))

            if name == "LightGBM":
                eval_result = model.evals_result_
                metric = list(eval_result['training'].keys())[0]
                plt.plot(eval_result['training'][metric], label='Train')
                plt.plot(eval_result['valid_1'][metric], label='Validation')
                plt.ylabel('Loss')

            elif name == "XGBoost":
                eval_result = model.evals_result()
                metric = list(eval_result['validation_0'].keys())[0]
                plt.plot(eval_result['validation_0'][metric], label='Train (validation_0)')
                plt.plot(eval_result['validation_1'][metric], label='Validation (validation_1)')
                plt.ylabel('Loss')

            elif name == "CatBoost":
                eval_result = model.get_evals_result()
                metric = list(eval_result['learn'].keys())[0]
                plt.plot(eval_result['learn'][metric], label='Train')
                plt.plot(eval_result['validation'][metric], label='Validation')
                plt.ylabel('Loss')

            plt.title(f'{name} Learning Curve')
            plt.xlabel('Iterations')
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(png_dir, f'C1_{name}_Learning_Curve.png'),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

        except Exception as e:
            print(f"無法繪製 {name} 的學習曲線: {e}")

    # =========================================================
    # (B) 訓練神經網路模型 (MLP)
    # =========================================================

    # print("\n正在訓練 Neural Network (MLP)...")

    # nn_model = build_nn_model(X_train.shape[1])

    # early_stop = callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=10,
    #     restore_best_weights=True
    # )

    # start_nn = time.time()

    # history = nn_model.fit(
    #     X_train.toarray(),
    #     y_train,
    #     validation_data=(X_val.toarray(), y_val),
    #     epochs=100,
    #     batch_size=32,
    #     callbacks=[early_stop],
    #     verbose=1
    # )

    # duration_nn = time.time() - start_nn

    # y_pred_nn = nn_model.predict(X_val.toarray()).flatten()

    # rmse_nn = np.sqrt(mean_squared_error(y_val, y_pred_nn))
    # mae_nn = mean_absolute_error(y_val, y_pred_nn)
    # r2_nn = r2_score(y_val, y_pred_nn)

    # y_val_inv_nn = pt_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    # y_pred_inv_nn = pt_y.inverse_transform(y_pred_nn.reshape(-1, 1)).flatten()

    # rmse_inv_nn = np.sqrt(mean_squared_error(y_val_inv_nn, y_pred_inv_nn))
    # mae_inv_nn = mean_absolute_error(y_val_inv_nn, y_pred_inv_nn)
    # r2_inv_nn = r2_score(y_val_inv_nn, y_pred_inv_nn)

    # mape_inv_nn = np.mean(
    #     np.abs(y_val_inv_nn - y_pred_inv_nn) / np.maximum(np.abs(y_val_inv_nn), 1)
    # ) * 100

    # # 生成 Neural Network Error Map
    # generate_error_map(
    #     y_true_days=y_val_inv_nn,
    #     y_pred_days=y_pred_inv_nn,
    #     model_name="Neural_Network",
    #     output_dir=png_dir
    # )

    # res_nn = (
    #     f"[Neural Network] "
    #     f"PT Space - RMSE: {rmse_nn:.4f}, MAE: {mae_nn:.4f}, R2: {r2_nn:.4f} | "
    #     f"Real Space - RMSE: {rmse_inv_nn:.4f}, MAE: {mae_inv_nn:.4f}, "
    #     f"R2: {r2_inv_nn:.4f}, MAPE: {mape_inv_nn:.2f}% | "
    #     f"Time: {duration_nn:.2f}s\n"
    # )

    # print(res_nn.strip())
    # f_log.write(res_nn)

    # results.append({
    #     "Model": "Neural Network",
    #     "RMSE (PT)": rmse_nn,
    #     "MAE (PT)": mae_nn,
    #     "R2 (PT)": r2_nn,
    #     "RMSE (Days)": rmse_inv_nn,
    #     "MAE (Days)": mae_inv_nn,
    #     "R2 (Days)": r2_inv_nn,
    #     "MAPE(%)": mape_inv_nn
    # })

    # nn_model.save('C1_NN_Baseline_Model.h5')

    # try:
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(history.history['loss'], label='Train Loss (MSE)')
    #     plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    #     plt.title('Neural Network Learning Curve')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(
    #         os.path.join(png_dir, 'C1_NeuralNetwork_Learning_Curve.png'),
    #         dpi=300,
    #         bbox_inches='tight'
    #     )
    #     plt.close()

    # except Exception as e:
    #     print(f"無法繪製 Neural Network 的學習曲線: {e}")

    # =========================================================
    # (C) Repeated K-Fold Cross Validation
    # 根據指示，略過 Repeated K-Fold CV
#     # =========================================================
# 
#     print("\n=====================================================================")
#     print("開始執行 Repeated K-Fold Cross Validation (n_splits=5, n_repeats=3)")
#     print("=====================================================================")
# 
#     try:
#         print("讀取全量特徵資料 (Full Data)...")
# 
#         X_full = sp.load_npz('C1_ML_Full_X.npz')
#         y_full = np.load('C1_ML_Full_y.npy')
# 
#         rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
# 
#         cv_records = {name: [] for name in models_dict.keys()}
#         cv_records["Neural Network"] = []
# 
#         f_log.write("\n=== K-Fold Cross Validation (5 Splits, 3 Repeats) ===\n")
# 
#         fold_idx = 1
# 
#         for train_ix, val_ix in rkf.split(X_full):
#             print(f"\n[CV Run {fold_idx}/15]")
# 
#             X_tr_cv, X_va_cv = X_full[train_ix], X_full[val_ix]
#             y_tr_cv, y_va_cv = y_full[train_ix], y_full[val_ix]
# 
#             y_va_inv_cv = pt_y.inverse_transform(
#                 y_va_cv.reshape(-1, 1)
#             ).flatten()
# 
#             for name, model_cls in models_dict.items():
#                 model_cv = clone(model_cls)
# 
#                 if name == "CatBoost":
#                     model_cv.fit(
#                         X_tr_cv,
#                         y_tr_cv,
#                         eval_set=(X_va_cv, y_va_cv),
#                         early_stopping_rounds=50,
#                         verbose=0
#                     )
# 
#                 elif name == "LightGBM":
#                     model_cv.fit(
#                         X_tr_cv,
#                         y_tr_cv,
#                         eval_set=[(X_va_cv, y_va_cv)]
#                     )
# 
#                 else:
#                     model_cv.fit(
#                         X_tr_cv,
#                         y_tr_cv,
#                         eval_set=[(X_va_cv, y_va_cv)],
#                         verbose=0
#                     )
# 
#                 y_pred_cv = model_cv.predict(X_va_cv)
#                 y_pred_inv_cv = pt_y.inverse_transform(
#                     y_pred_cv.reshape(-1, 1)
#                 ).flatten()
# 
#                 rmse_inv_cv = np.sqrt(
#                     mean_squared_error(y_va_inv_cv, y_pred_inv_cv)
#                 )
# 
#                 mae_inv_cv = mean_absolute_error(
#                     y_va_inv_cv, y_pred_inv_cv
#                 )
# 
#                 mape_inv_cv = np.mean(
#                     np.abs(y_va_inv_cv - y_pred_inv_cv)
#                     / np.maximum(np.abs(y_va_inv_cv), 1)
#                 ) * 100
# 
#                 cv_records[name].append({
#                     "RMSE": rmse_inv_cv,
#                     "MAE": mae_inv_cv,
#                     "MAPE": mape_inv_cv
#                 })
# 
#             nn_model_cv = build_nn_model(X_full.shape[1])
# 
#             early_stop_cv = callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 patience=10,
#                 restore_best_weights=True
#             )
# 
#             nn_model_cv.fit(
#                 X_tr_cv.toarray(),
#                 y_tr_cv,
#                 validation_data=(X_va_cv.toarray(), y_va_cv),
#                 epochs=100,
#                 batch_size=32,
#                 callbacks=[early_stop_cv],
#                 verbose=0
#             )
# 
#             y_pred_nn_cv = nn_model_cv.predict(
#                 X_va_cv.toarray(),
#                 verbose=0
#             ).flatten()
# 
#             y_pred_inv_nn_cv = pt_y.inverse_transform(
#                 y_pred_nn_cv.reshape(-1, 1)
#             ).flatten()
# 
#             rmse_inv_nn_cv = np.sqrt(
#                 mean_squared_error(y_va_inv_cv, y_pred_inv_nn_cv)
#             )
# 
#             mae_inv_nn_cv = mean_absolute_error(
#                 y_va_inv_cv, y_pred_inv_nn_cv
#             )
# 
#             mape_inv_nn_cv = np.mean(
#                 np.abs(y_va_inv_cv - y_pred_inv_nn_cv)
#                 / np.maximum(np.abs(y_va_inv_cv), 1)
#             ) * 100
# 
#             cv_records["Neural Network"].append({
#                 "RMSE": rmse_inv_nn_cv,
#                 "MAE": mae_inv_nn_cv,
#                 "MAPE": mape_inv_nn_cv
#             })
# 
#             fold_idx += 1
# 
#         print("\n================= K-Fold CV Results =================")
# 
#         cv_summary = []
# 
#         for m_name, metrics_list in cv_records.items():
#             df_metrics = pd.DataFrame(metrics_list)
#             mean_metrics = df_metrics.mean()
#             std_metrics = df_metrics.std()
# 
#             res_str = (
#                 f"[{m_name}] CV Mean ± Std | "
#                 f"RMSE: {mean_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}, "
#                 f"MAE: {mean_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f}, "
#                 f"MAPE: {mean_metrics['MAPE']:.2f}% ± {std_metrics['MAPE']:.2f}%\n"
#             )
# 
#             print(res_str.strip())
#             f_log.write(res_str)
# 
#             cv_summary.append({
#                 "Model": m_name,
#                 "CV_RMSE_Mean": mean_metrics['RMSE'],
#                 "CV_RMSE_Std": std_metrics['RMSE'],
#                 "CV_MAE_Mean": mean_metrics['MAE'],
#                 "CV_MAE_Std": std_metrics['MAE'],
#                 "CV_MAPE_Mean(%)": mean_metrics['MAPE'],
#                 "CV_MAPE_Std(%)": std_metrics['MAPE']
#             })
# 
#         pd.DataFrame(cv_summary).to_csv(
#             'C1_Model_CV_Summary.csv',
#             index=False
#         )
# 
#         print("K-Fold 結果已儲存為 C1_Model_CV_Summary.csv")
# 
#     except Exception as e:
#         print(f"嘗試執行 K-Fold CV 失敗: {e}")
# 
# 
# # =========================================================
# 5. 產出報告
# =========================================================

print("\n正在產出訓練報表...")

if importance_list:
    all_importance = pd.concat(importance_list, axis=1)
    all_importance.to_csv('C1_Feature_Importance_Report.csv')

pd.DataFrame(results).to_csv(
    'C1_Model_Comparison_Summary.csv',
    index=False
)

print("所有模型 (Tree & NN) 訓練完成！請查看產出的 CSV 報告與模型檔案。")