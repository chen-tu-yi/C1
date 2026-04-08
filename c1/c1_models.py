import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import joblib
import os
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning, message='.*valid feature names.*')

png_dir = r"C:\local_文件\專題\png\c1"
os.makedirs(png_dir, exist_ok=True)

# Tree Models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# NN Models
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 1. 讀取與資料清洗
# ==============================================================================
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

# 2. 定義 Tree 型模型
# ==============================================================================
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

# 3. 定義神經網路結構 (MLP)
# ==============================================================================
def build_nn_model(input_shape):
    model = models.Sequential([
        # 第一層：輸入層 + 隱藏層 (64 顆神經元)
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.2),  # 隨機捨棄 20% 神經元，防止過擬合
        
        # 第二層：隱藏層 (32 顆神經元)
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # 第三層：隱藏層 (16 顆神經元)
        layers.Dense(16, activation='relu'),
        
        # 輸出層：回歸問題只需 1 顆神經元，且不加激發函數
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 4. 執行訓練與 Log 紀錄
# ==============================================================================
results = []
importance_list = []

with open("training_log.txt", "w", encoding="utf-8") as f_log:
    f_log.write("=== C1 採購延遲預測 訓練實驗紀錄 ===\n")
    f_log.write(f"執行時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f_log.write(f"樣本數: {total_samples}, 特徵數: {X_train.shape[1]}\n\n")

    # (A) 訓練 Tree 型模型
    for name, model in models_dict.items():
        print(f"\n正在訓練 {name}...")
        start = time.time()
        
        # 訓練模型並加入驗證集監控
        if name == "CatBoost":
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        else:
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
            
        # 儲存 XGBoost 模型供 shortage_to_model.py 推論使用
        if name == "XGBoost":
            joblib.dump(model, 'XGBoost_Model.joblib')
            print(f"XGBoost 模型已儲存為 XGBoost_Model.joblib")
        
        duration = time.time() - start
        y_pred = model.predict(X_val)
        
        # 1. PT 空間下的指標
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # 2. 真實天數空間的反正規化 (Inverse Transform)
        y_val_inv = pt_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_inv = pt_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # 容錯處理：預測出負天數或極小數值時，設下限為 0 天 (此處已移除)
        # 1. 不要強迫變 0，保留原始天數 (包含負數)
        
        rmse_inv = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))
        mae_inv = mean_absolute_error(y_val_inv, y_pred_inv)
        r2_inv = r2_score(y_val_inv, y_pred_inv)
        
        # 2. 計算比例時，分母取絕對值，且最少算 1 天 (避免 0 天或極小值)
        # 採用 Adjusted MAPE
        mape_inv = np.mean(np.abs(y_val_inv - y_pred_inv) / np.maximum(np.abs(y_val_inv), 1)) * 100
        
        # --- WAPE (Weighted Absolute Percentage Error) 備註 ---
        # 1. 計算分子：所有訂單絕對誤差的總和
        # sum_abs_error = np.sum(np.abs(y_val_inv - y_pred_inv))
        # 2. 計算分母：所有實際進貨天數的絕對值總和
        # sum_abs_actual = np.sum(np.abs(y_val_inv))
        # 3. 相除得到 WAPE (加上防止分母為 0 的防呆機制)
        # wape_inv = (sum_abs_error / sum_abs_actual * 100) if sum_abs_actual != 0 else 0
        
        res = f"[{name}] PT Space - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f} | " \
              f"Real Space - RMSE: {rmse_inv:.4f}, MAE: {mae_inv:.4f}, R2: {r2_inv:.4f}, MAPE: {mape_inv:.2%} | Time: {duration:.2f}s\n"
        print(res.strip())
        f_log.write(res)
        results.append({
            "Model": name, 
            "RMSE (PT)": rmse, "MAE (PT)": mae, "R2 (PT)": r2,
            "RMSE (Days)": rmse_inv, "MAE (Days)": mae_inv, "R2 (Days)": r2_inv, "MAPE(%)": mape_inv
            # , "WAPE(%)": wape_inv
        })
        
        # 提取特徵重要性 (Feature Importance)
        feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        importance_list.append(feat_imp.to_frame(name=name))
        
        # 3. 繪製學習曲線
        try:
            plt.figure(figsize=(10, 6))
            if name == "LightGBM":
                res = model.evals_result_
                metric = list(res['training'].keys())[0]
                plt.plot(res['training'][metric], label='Train')
                plt.plot(res['valid_1'][metric], label='Validation')
                plt.ylabel('Loss')
            elif name == "XGBoost":
                res = model.evals_result()
                metric = list(res['validation_0'].keys())[0]
                plt.plot(res['validation_0'][metric], label='Train (validation_0)')
                plt.plot(res['validation_1'][metric], label='Validation (validation_1)')
                plt.ylabel('Loss')
            elif name == "CatBoost":
                res = model.get_evals_result()
                metric = list(res['learn'].keys())[0]
                plt.plot(res['learn'][metric], label='Train')
                plt.plot(res['validation'][metric], label='Validation')
                plt.ylabel('Loss')
            
            plt.title(f'{name} Learning Curve')
            plt.xlabel('Iterations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(png_dir, f'C1_{name}_Learning_Curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"無法繪製 {name} 的學習曲線: {e}")

    # (B) 訓練神經網路模型 (MLP)
    print("\n正在訓練 Neural Network (MLP)...")
    nn_model = build_nn_model(X_train.shape[1])
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    start_nn = time.time()
    
    # 由於經過降維且記憶體不再受 Pandas 影響，可以直接呼叫 .toarray() 餵給 Keras
    history = nn_model.fit(
        X_train.toarray(), y_train,
        validation_data=(X_val.toarray(), y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    duration_nn = time.time() - start_nn
    y_pred_nn = nn_model.predict(X_val.toarray()).flatten()
    
    # 1. PT 空間下的指標
    rmse_nn = np.sqrt(mean_squared_error(y_val, y_pred_nn))
    mae_nn = mean_absolute_error(y_val, y_pred_nn)
    r2_nn = r2_score(y_val, y_pred_nn)
    
    # 2. 真實天數空間的反正規化 (Inverse Transform)
    y_val_inv_nn = pt_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_inv_nn = pt_y.inverse_transform(y_pred_nn.reshape(-1, 1)).flatten()
    # 不要強迫變 0，保留原始天數 (包含負數)
    
    rmse_inv_nn = np.sqrt(mean_squared_error(y_val_inv_nn, y_pred_inv_nn))
    mae_inv_nn = mean_absolute_error(y_val_inv_nn, y_pred_inv_nn)
    r2_inv_nn = r2_score(y_val_inv_nn, y_pred_inv_nn)
    
    # 計算 Adjusted MAPE
    mape_inv_nn = np.mean(np.abs(y_val_inv_nn - y_pred_inv_nn) / np.maximum(np.abs(y_val_inv_nn), 1)) * 100
    
    # --- WAPE (Weighted Absolute Percentage Error) 備註 ---
    # 1. 計算分子：所有訂單絕對誤差的總和
    # sum_abs_error_nn = np.sum(np.abs(y_val_inv_nn - y_pred_inv_nn))
    # 2. 計算分母：所有實際進貨天數的絕對值總和
    # sum_abs_actual_nn = np.sum(np.abs(y_val_inv_nn))
    # 3. 相除得到 WAPE (加上防止分母為 0 的防呆機制)
    # wape_inv_nn = (sum_abs_error_nn / sum_abs_actual_nn * 100) if sum_abs_actual_nn != 0 else 0
    
    res_nn = f"[Neural Network] PT Space - RMSE: {rmse_nn:.4f}, MAE: {mae_nn:.4f}, R2: {r2_nn:.4f} | " \
             f"Real Space - RMSE: {rmse_inv_nn:.4f}, MAE: {mae_inv_nn:.4f}, R2: {r2_inv_nn:.4f}, MAPE: {mape_inv_nn:.2%} | Time: {duration_nn:.2f}s\n"
    print(res_nn.strip())
    f_log.write(res_nn)
    results.append({
        "Model": "Neural Network", 
        "RMSE (PT)": rmse_nn, "MAE (PT)": mae_nn, "R2 (PT)": r2_nn,
        "RMSE (Days)": rmse_inv_nn, "MAE (Days)": mae_inv_nn, "R2 (Days)": r2_inv_nn, "MAPE(%)": mape_inv_nn
        # , "WAPE(%)": wape_inv_nn
    })
    
    # NN 儲存為 .h5 模型
    nn_model.save('C1_NN_Baseline_Model.h5')
    
    # 繪製神經網路學習曲線
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss (MSE)')
        plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
        plt.title('Neural Network Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(png_dir, 'C1_NeuralNetwork_Learning_Curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"無法繪製 Neural Network 的學習曲線: {e}")

    # (C) 進行 Repeated K-Fold Cross Validation
    print("\n=====================================================================")
    print("開始執行 Repeated K-Fold Cross Validation (n_splits=5, n_repeats=3)")
    print("=====================================================================")
    
    try:
        print("讀取全量特徵資料 (Full Data)...")
        X_full = sp.load_npz('C1_ML_Full_X.npz')
        y_full = np.load('C1_ML_Full_y.npy')
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        cv_records = {name: [] for name in models_dict.keys()}
        cv_records["Neural Network"] = []
        
        f_log.write("\n=== K-Fold Cross Validation (5 Splits, 3 Repeats) ===\n")
        
        fold_idx = 1
        for train_ix, val_ix in rkf.split(X_full):
            print(f"\n[CV Run {fold_idx}/15]")
            X_tr_cv, X_va_cv = X_full[train_ix], X_full[val_ix]
            y_tr_cv, y_va_cv = y_full[train_ix], y_full[val_ix]
            
            # 暫存真值的天數轉換
            y_va_inv_cv = pt_y.inverse_transform(y_va_cv.reshape(-1, 1)).flatten()
            
            # 訓練 Tree Models
            for name, model_cls in models_dict.items():
                model_cv = clone(model_cls)
                if name == "CatBoost":
                    model_cv.fit(X_tr_cv, y_tr_cv, eval_set=(X_va_cv, y_va_cv), early_stopping_rounds=50, verbose=0)
                elif name == "LightGBM":
                    model_cv.fit(X_tr_cv, y_tr_cv, eval_set=[(X_va_cv, y_va_cv)])
                else:
                    model_cv.fit(X_tr_cv, y_tr_cv, eval_set=[(X_va_cv, y_va_cv)], verbose=0)
                    
                y_pred_cv = model_cv.predict(X_va_cv)
                y_pred_inv_cv = pt_y.inverse_transform(y_pred_cv.reshape(-1, 1)).flatten()
                
                rmse_inv_cv = np.sqrt(mean_squared_error(y_va_inv_cv, y_pred_inv_cv))
                mae_inv_cv = mean_absolute_error(y_va_inv_cv, y_pred_inv_cv)
                mape_inv_cv = np.mean(np.abs(y_va_inv_cv - y_pred_inv_cv) / np.maximum(np.abs(y_va_inv_cv), 1)) * 100
                cv_records[name].append({"RMSE": rmse_inv_cv, "MAE": mae_inv_cv, "MAPE": mape_inv_cv})
                
            # 訓練 Neural Network
            nn_model_cv = build_nn_model(X_full.shape[1])
            early_stop_cv = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            nn_model_cv.fit(
                X_tr_cv.toarray(), y_tr_cv,
                validation_data=(X_va_cv.toarray(), y_va_cv),
                epochs=100, batch_size=32, callbacks=[early_stop_cv], verbose=0
            )
            y_pred_nn_cv = nn_model_cv.predict(X_va_cv.toarray(), verbose=0).flatten()
            y_pred_inv_nn_cv = pt_y.inverse_transform(y_pred_nn_cv.reshape(-1, 1)).flatten()
            
            rmse_inv_nn_cv = np.sqrt(mean_squared_error(y_va_inv_cv, y_pred_inv_nn_cv))
            mae_inv_nn_cv = mean_absolute_error(y_va_inv_cv, y_pred_inv_nn_cv)
            mape_inv_nn_cv = np.mean(np.abs(y_va_inv_cv - y_pred_inv_nn_cv) / np.maximum(np.abs(y_va_inv_cv), 1)) * 100
            cv_records["Neural Network"].append({"RMSE": rmse_inv_nn_cv, "MAE": mae_inv_nn_cv, "MAPE": mape_inv_nn_cv})
            
            fold_idx += 1
            
        print("\n================= K-Fold CV Results =================")
        cv_summary = []
        for m_name, metrics_list in cv_records.items():
            df_metrics = pd.DataFrame(metrics_list)
            mean_metrics = df_metrics.mean()
            std_metrics = df_metrics.std()
            
            res_str = (f"[{m_name}] CV Mean ± Std | "
                       f"RMSE: {mean_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}, "
                       f"MAE: {mean_metrics['MAE']:.4f} ± {std_metrics['MAE']:.4f}, "
                       f"MAPE: {mean_metrics['MAPE']:.2f}% ± {std_metrics['MAPE']:.2f}%\n")
            print(res_str.strip())
            f_log.write(res_str)
            
            cv_summary.append({
                "Model": m_name,
                "CV_RMSE_Mean": mean_metrics['RMSE'], "CV_RMSE_Std": std_metrics['RMSE'],
                "CV_MAE_Mean": mean_metrics['MAE'], "CV_MAE_Std": std_metrics['MAE'],
                "CV_MAPE_Mean(%)": mean_metrics['MAPE'], "CV_MAPE_Std(%)": std_metrics['MAPE']
            })
            
        pd.DataFrame(cv_summary).to_csv('C1_Model_CV_Summary.csv', index=False)
        print("K-Fold 結果已儲存為 C1_Model_CV_Summary.csv")
    except Exception as e:
        print(f"嘗試執行 K-Fold CV 失敗: {e}")

# 5. 產出報告
# ==============================================================================
print("\n正在產出訓練報表...")
if importance_list:
    all_importance = pd.concat(importance_list, axis=1)
    all_importance.to_csv('C1_Feature_Importance_Report.csv')

pd.DataFrame(results).to_csv('C1_Model_Comparison_Summary.csv', index=False)

print("所有模型 (Tree & NN) 訓練完成！請查看產出的 CSV 報告與模型檔案。")
