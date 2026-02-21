import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, concatenate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==== Baseline models (scikit-learn) ====
def build_lgbm_classifier():
    return LGBMClassifier(learning_rate=0.1, max_depth=20, n_estimators=100, random_state=42, verbose=-1)

def build_rf_classifier():
    return RandomForestClassifier(max_depth=None, n_estimators=200, random_state=42)

def build_et_classifier():
    """Extra Trees Classifier (baseline tree model)."""
    return ExtraTreesClassifier(max_depth=20, n_estimators=200, random_state=42)

# ==== XGBoost classifier ====
def build_xgb_classifier():
    """XGBoost Classifier (baseline tree model)."""
    return XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

# ==== CatBoost classifier ====
def build_catboost_classifier():
    """CatBoost Classifier (baseline tree model)."""
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=0
    )


# ==== DNN meta-model ====
def build_dnn_meta(input_dim: int, n_classes: int = 1) -> tf.keras.Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    if n_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="DNN_Meta")

# ==== RNN meta-model (LSTM) – GIẢI THÍCH ====
def build_rnn_meta(input_dim: int, n_classes: int = 1) -> tf.keras.Model:
    """
    RNN meta-learner dùng LSTM cho stacking ensemble.

    LÝ DO GIỮ LẠI:
    - Dữ liệu đầu vào cho meta-learner là vector xác suất từ các base models
      (thường có kích thước nhỏ, ví dụ 3 models → 3 chiều).
    - Mặc dù dữ liệu này không có thứ tự thời gian, nhưng LSTM có thể học được
      các tương tác phi tuyến phức tạp giữa các đầu ra của base models.
    - Một số nghiên cứu cho thấy việc coi các base models như một chuỗi
      và áp dụng RNN có thể cải thiện hiệu suất so với hồi quy logistic truyền thống
      (Wolpert 1992, stacking generalization; và các biến thể hiện đại).
    - Trong pipeline này, RNN meta-learner được giữ như một phương án thử nghiệm
      để so sánh với DNN và LR. Nếu không cải thiện, có thể loại bỏ sau này.

    LƯU Ý: Input cần reshape về (batch, timesteps, features). Với stacking OOF,
    ta có thể reshape (n_samples, 1, n_features).
    """
    inputs = Input(shape=(None, input_dim))
    x = LSTM(64, activation='relu', return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(32, activation='relu', return_sequences=False)(x)
    x = Dropout(0.2)(x)
    if n_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(n_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="RNN_Meta")

# ==== Multitask DNN cho bài toán chính ====
def build_multitask_dnn(input_dim: int) -> tf.keras.Model:
    """Multitask DNN với 3 đầu ra: churn (binary), credit score class (3 classes), high balance flag (binary)."""
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    # Shared layers
    shared = Dense(32, activation='relu')(x)
    # Task-specific heads
    churn_out = Dense(1, activation='sigmoid', name='churn')(shared)
    score_out = Dense(3, activation='softmax', name='score')(shared)
    balance_out = Dense(1, activation='sigmoid', name='balance')(shared)
    model = Model(inputs=inputs, outputs=[churn_out, score_out, balance_out])
    return model