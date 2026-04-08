import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import os
from xgboost import XGBRegressor

def train():
    BASE_DIR = r'C:\local_file\專題\c1'
    MODEL_DIR = r'C:\local_file\專題\model'
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("讀取 Sparse Matrix 特徵資料...")
    X_train = sp.load_npz(os.path.join(BASE_DIR, 'C1_ML_Training_X.npz'))
    y_train = np.load(os.path.join(BASE_DIR, 'C1_ML_Training_y.npy'))
    X_val = sp.load_npz(os.path.join(BASE_DIR, 'C1_ML_Test_X.npz'))
    y_val = np.load(os.path.join(BASE_DIR, 'C1_ML_Test_y.npy'))

    print("訓練 XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        random_state=42, 
        n_jobs=2,
        eval_metric='rmse'
    )

    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=50)

    model_path = os.path.join(MODEL_DIR, 'XGBoost_Model.joblib')
    joblib.dump(xgb_model, model_path)
    print(f"XGBoost 模型已儲存至 {model_path}")

if __name__ == '__main__':
    train()
