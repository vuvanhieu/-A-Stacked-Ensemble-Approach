import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from data_ultils import apply_smote_in_fold
from evaluations import plot_all, export_compare_results_to_csv, plot_model_comparison
from configs import N_FOLDS, N_REPEATS
from configs import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
import joblib
import json


def cross_validate_base_models(
    X, y,
    base_model_builders,
    n_splits=5,
    random_state=42,
    task_name="churn",
    use_smote=True,
    base_output_dir=None
):
    """
    Thực hiện k-fold cross-validation cho mỗi base model.
    Trả về:
        - best_models: dict {model_name: best_model} – mô hình từ fold có validation score cao nhất
        - cv_records: list of dict – kết quả từng fold cho từng model (để tính mean/std)
        - oof_preds: dict {model_name: np.array} – OOF predictions (dùng để tạo meta-features)
    """
    if N_REPEATS > 1:
        kf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=random_state)
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)

    base_names = ["LGBM", "RF", "ET", "XGB", "CATBOOST"]
    # Đồng bộ: builders phải có đúng 5 phần tử tương ứng base_names
    if len(base_model_builders) == 5:
        builders = base_model_builders
    else:
        raise ValueError("base_model_builders phải có đúng 5 phần tử: LGBM, RF, ET, XGB, CATBOOST")

    # Lưu tất cả models từ các fold (để chọn best sau)
    fold_models = {name: [] for name in base_names}
    fold_scores = {name: [] for name in base_names}
    # OOF predictions: shape (n_samples, n_models)
    oof_preds = np.zeros((X.shape[0], len(base_names)))
    # Validation metrics cho mỗi fold
    cv_records = []

    # Lấy tên cột từ DataFrame gốc nếu có
    feature_cols = None
    if isinstance(X, pd.DataFrame):
        feature_cols = X.columns.tolist()
    else:
        # Nếu X là numpy array, lấy từ df_tree nếu truyền vào
        try:
            from main import preprocess_tree_based
            _, _, df_tree = preprocess_tree_based(pd.read_csv('Bank_Customer_Churn.csv'))
            feature_cols = [col for col in df_tree.columns if col not in ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'CreditScoreClass', 'HighBalanceFlag']]
        except Exception:
            feature_cols = None


    if base_output_dir is None:
        # fallback giữ nguyên hành vi cũ nếu không truyền
        from configs import BACKUP_DIRNAME, BASELINE_DIRNAME
        base_output_dir = os.path.join(BACKUP_DIRNAME, BASELINE_DIRNAME)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Chuyển về DataFrame nếu chưa có tên cột
        if feature_cols is not None and not isinstance(X_tr, pd.DataFrame):
            X_tr = pd.DataFrame(X_tr, columns=feature_cols)
            X_val = pd.DataFrame(X_val, columns=feature_cols)

        if use_smote:
            try:
                X_tr, y_tr = apply_smote_in_fold(X_tr, y_tr)
            except Exception as e:
                print(f"SMOTE failed in fold {fold}: {e}, using original data")

        for i, build_model in enumerate(builders):
            model = build_model()
            model.fit(X_tr, y_tr)

            # Dự đoán trên validation
            if hasattr(model, "predict_proba"):
                if task_name == "score":  # multiclass
                    y_pred_proba = model.predict_proba(X_val)  # shape (n_val, n_classes)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:  # binary
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_val)
                y_pred_proba = y_pred

            # Tính metric phù hợp
            if task_name == "score":
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                score = f1  # dùng F1-weighted để chọn best fold
            else:  # churn, balance (binary)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                score = f1  # dùng F1 cho binary

            # Lưu OOF predictions chỉ cho binary (churn, balance)
            if task_name != "score":
                oof_preds[val_idx, i] = y_pred_proba

            # Lưu model và score
            fold_models[base_names[i]].append(model)
            fold_scores[base_names[i]].append(score)

            # Lưu metrics/plots cho từng fold
            # Tạo thư mục: OUTPUT/{run_id}/Baselines/{model_name}/fold_{fold+1}
            fold_dir = os.path.join(base_output_dir, base_names[i], f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)

            # Lưu model
            model_path = os.path.join(fold_dir, f"{base_names[i]}_fold{fold+1}.joblib")
            joblib.dump(model, model_path)

            # Lưu metrics
            metrics = {
                "Model": base_names[i],
                "Fold": fold,
                "Accuracy": acc,
                "F1-score": f1,
                "Precision": prec,
                "Recall": rec
            }
            metrics_path = os.path.join(fold_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Lưu các plot (confusion matrix, ROC, PR, AUC)
            try:
                plot_all(y_val, y_pred, y_pred_proba if hasattr(model, "predict_proba") else y_pred, task_name, fold_dir, model_name=base_names[i])
            except Exception as e:
                print(f"Plotting error for {base_names[i]} fold {fold+1}: {e}")

            cv_records.append(metrics)

    # Chọn best model cho mỗi loại (theo score cao nhất)
    best_models = {}
    for name in base_names:
        scores = fold_scores[name]
        best_idx = np.argmax(scores)
        best_models[name] = fold_models[name][best_idx]
        print(f"Best {name} fold: {best_idx+1} with score {scores[best_idx]:.4f}")

    return best_models, cv_records, oof_preds




def train_meta_model(
    meta_features_train, y_train,
    meta_features_test, y_test,
    meta_model_builder,
    meta_model_type="sklearn",  # "sklearn", "dnn", "rnn"
    task_name="churn",
    save_dir=None,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE
):
    """
    Huấn luyện meta‑learner trên meta‑features.
    Trả về: y_pred_test, y_proba_test, model, metrics.
    """
    history = None
    if meta_model_type == "sklearn":
        meta_model = meta_model_builder()
        meta_model.fit(meta_features_train, y_train)
        y_pred_test = meta_model.predict(meta_features_test)
        if hasattr(meta_model, "predict_proba"):
            y_proba_test = meta_model.predict_proba(meta_features_test)
            if task_name == "score":
                # multiclass: proba shape (n, n_classes)
                pass
            else:
                y_proba_test = y_proba_test[:, 1]
        else:
            y_proba_test = y_pred_test
        # sklearn không có history
        history = None
    elif meta_model_type in ["dnn", "rnn"]:
        meta_model = meta_model_builder()
        # Compile phù hợp với task
        if task_name == "score":
            loss = 'categorical_crossentropy'
            output_activation = 'softmax'
            # y đã được one-hot hóa từ main.py
        else:
            loss = 'binary_crossentropy'
            output_activation = 'sigmoid'
        meta_model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        # Reshape nếu RNN
        if meta_model_type == "rnn":
            meta_features_train = meta_features_train.reshape((meta_features_train.shape[0], 1, meta_features_train.shape[1]))
            meta_features_test = meta_features_test.reshape((meta_features_test.shape[0], 1, meta_features_test.shape[1]))

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        history = meta_model.fit(
            meta_features_train, y_train,
            validation_split=0.2,
            epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop]
        )
        y_proba_test = meta_model.predict(meta_features_test)
        if task_name == "score":
            y_pred_test = np.argmax(y_proba_test, axis=1)
        else:
            y_pred_test = (y_proba_test > 0.5).astype(int).flatten()
            y_proba_test = y_proba_test.flatten()

    # Tính metrics
    if task_name == "score" and meta_model_type in ["dnn", "rnn"]:
        y_test_eval = np.argmax(y_test, axis=1)
    else:
        y_test_eval = y_test
    if task_name == "score":
        acc = accuracy_score(y_test_eval, y_pred_test)
        f1 = f1_score(y_test_eval, y_pred_test, average='weighted', zero_division=0)
        prec = precision_score(y_test_eval, y_pred_test, average='weighted', zero_division=0)
        rec = recall_score(y_test_eval, y_pred_test, average='weighted', zero_division=0)
    else:
        acc = accuracy_score(y_test_eval, y_pred_test)
        f1 = f1_score(y_test_eval, y_pred_test, zero_division=0)
        prec = precision_score(y_test_eval, y_pred_test, zero_division=0)
        rec = recall_score(y_test_eval, y_pred_test, zero_division=0)

    if save_dir:
        plot_all(y_test, y_pred_test, y_proba_test, task_name, save_dir,
                 model_name=f"Meta_{meta_model.__class__.__name__}")

    return y_pred_test, y_proba_test, meta_model, {"Accuracy": acc, "F1-score": f1, "Precision": prec, "Recall": rec}, history


import os
import numpy as np
import tensorflow as tf
from evaluations import plot_all

def train_multitask_model(
    model,
    X_train, y_train_dict,
    X_val, y_val_dict,
    X_test, y_test_dict,
    epochs=150,
    batch_size=32,
    save_path=None,
    save_dir=None,
    model_name="Multitask DNN"
):
    """
    Huấn luyện mô hình đa nhiệm (3 đầu ra: churn, score, balance) với dữ liệu đã cân bằng.
    
    Args:
        model: Keras model (build_multitask_dnn)
        X_train, y_train_dict: dict với keys 'churn', 'score', 'balance'
        X_val, y_val_dict: dict validation
        X_test, y_test_dict: dict test
        epochs: số epoch
        batch_size: batch size
        save_path: đường dẫn lưu model .h5 (tốt nhất)
        save_dir: thư mục lưu các plot
        model_name: tên hiển thị
    
    Returns:
        history: lịch sử huấn luyện
        test_metrics: list of dict cho mỗi task (churn, score, balance)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Compile model với các loss và metrics phù hợp cho từng đầu ra
    model.compile(
        optimizer='adam',
        loss={
            'churn': 'binary_crossentropy',
            'score': 'categorical_crossentropy',
            'balance': 'binary_crossentropy'
        },
        metrics={
            'churn': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
            'score': ['accuracy'],
            'balance': ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
        }
    )

    # Callbacks
    callbacks = []
    if save_path:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path, monitor='val_loss', save_best_only=True, verbose=0
        )
        callbacks.append(checkpoint)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    callbacks.append(early_stop)

    # Huấn luyện
    history = model.fit(
        X_train,
        {'churn': y_train_dict['churn'], 'score': y_train_dict['score'], 'balance': y_train_dict['balance']},
        validation_data=(
            X_val,
            {'churn': y_val_dict['churn'], 'score': y_val_dict['score'], 'balance': y_val_dict['balance']}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Đánh giá trên test set
    test_results = model.evaluate(
        X_test,
        {'churn': y_test_dict['churn'], 'score': y_test_dict['score'], 'balance': y_test_dict['balance']},
        verbose=0
    )

    # Lấy tên metrics từ model (thứ tự chính xác)
    metric_names = model.metrics_names
    results_dict = dict(zip(metric_names, test_results))

    # Tính metrics cho từng task trên test set
    test_metrics = []
    for i, task in enumerate(['churn', 'score', 'balance']):
        if task == 'score':
            y_true = y_test_dict[task]
            # Nếu y_true là one-hot, chuyển về vector nhãn
            if len(y_true.shape) == 2 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(model.predict(X_test, verbose=0)[i], axis=1)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            y_true = y_test_dict[task]
            y_pred = (model.predict(X_test, verbose=0)[i] > 0.5).astype(int).flatten()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
        test_metrics.append({
            "Task": task,
            "Accuracy": acc,
            "F1-score": f1,
            "Precision": prec,
            "Recall": rec,
            "Epochs": epochs,
            "Batch Size": batch_size
        })
    return history, test_metrics
