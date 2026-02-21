
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.neighbors import NearestNeighbors
import copy

# ===================== PREPROCESSING PIPELINES =====================
def preprocess_tree_based(df):
    """
    Pipeline dành cho tree-based models:
    - Label encode cho Geography, Gender
    - Tạo CreditScoreClass và HighBalanceFlag (nhưng không dùng làm feature)
    - KHÔNG scale, KHÔNG one-hot
    Trả về: X (numpy array), y_dict (dict), df_full (đã encode)
    """
    df = copy.deepcopy(df)
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['CreditScoreClass'] = pd.cut(
        df['CreditScore'], bins=[0, 580, 700, 850], labels=[0, 1, 2]
    ).astype(int)
    df['HighBalanceFlag'] = (df['Balance'] > 100000).astype(int)
    # Loại bỏ các cột không dùng cho mô hình tree-based
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'CreditScoreClass', 'HighBalanceFlag']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).values
    y_dict = {
        'churn': df['Exited'].values,
        'score': df['CreditScoreClass'].values,
        'balance': df['HighBalanceFlag'].values
    }
    return X, y_dict, df

def preprocess_neural(df):
    """
    Pipeline dành cho neural models:
    - Label encode Geography, Gender
    - Scale toàn bộ features (StandardScaler)
    - One-hot encoding cho CreditScoreClass (3 classes)
    - HighBalanceFlag giữ nguyên (binary)
    Trả về: X (scaled), y_dict (với score one-hot), df_full, scaler
    """
    df = copy.deepcopy(df)
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    df['Geography'] = le_geo.fit_transform(df['Geography'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['CreditScoreClass'] = pd.cut(
        df['CreditScore'], bins=[0, 580, 700, 850], labels=[0, 1, 2]
    ).astype(int)
    df['HighBalanceFlag'] = (df['Balance'] > 100000).astype(int)
        # Loại bỏ các cột không dùng cho mô hình
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'CreditScoreClass', 'HighBalanceFlag']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_dict = {
        'churn': df['Exited'].values,
        'score': to_categorical(df['CreditScoreClass'].values, num_classes=3),
        'balance': df['HighBalanceFlag'].values
    }
    return X_scaled, y_dict, df, scaler

def apply_smote_in_fold(X, y, random_state=42):
    """Áp dụng SMOTE-Tomek cho train fold (dùng trong stacking)."""
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res

def balance_labels(y_array, target_dist=None):
    """Undersampling để cân bằng các class."""
    df = pd.DataFrame({'label': y_array})
    counts = df['label'].value_counts()
    min_count = target_dist if target_dist else counts.min()
    balanced_idx = []
    for label in counts.index:
        label_idx = df[df['label'] == label].index
        sampled = np.random.choice(label_idx, size=min_count, replace=False)
        balanced_idx.extend(sampled)
    return np.array(balanced_idx)

def apply_balancing_all_tasks(X_train, y_train_dict):
    """
    Cân bằng toàn bộ tasks:
    - SMOTE-Tomek cho churn
    - Align score, balance
    - Undersampling để cân bằng score và balance
    """
    # Step 1: SMOTE-Tomek cho churn
    smt = SMOTETomek(random_state=42)
    X_res, y_churn_res = smt.fit_resample(X_train, y_train_dict['churn'])

    # Step 2: Align other labels
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
    _, indices = nbrs.kneighbors(X_res)
    indices = indices.flatten()

    y_score_res = y_train_dict['score'][indices]
    y_balance_res = y_train_dict['balance'][indices]

    # Step 3: Balance score (multiclass) và balance (binary)
    if y_score_res.ndim == 2:   # one-hot
        score_labels = np.argmax(y_score_res, axis=1)
    else:
        score_labels = y_score_res
    idx_score_bal = balance_labels(score_labels)
    idx_balance_bal = balance_labels(y_balance_res)

    # Intersect all indices
    final_idx = np.intersect1d(idx_score_bal, idx_balance_bal)

    return X_res[final_idx], {
        'churn': y_churn_res[final_idx],
        'score': y_score_res[final_idx],
        'balance': y_balance_res[final_idx]
    }

def scale_train_val_test(X_train, X_val=None, X_test=None):
    """Chuẩn hóa dựa trên X_train, trả về scaler và dữ liệu đã scale."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if X_val is not None else None
    X_test_s = scaler.transform(X_test) if X_test is not None else None
    return scaler, X_train_s, X_val_s, X_test_s