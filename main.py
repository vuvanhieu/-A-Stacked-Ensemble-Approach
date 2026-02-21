"""
main.py
Điều phối toàn bộ pipeline cho bài toán dự đoán churn khách hàng ngân hàng.

Các bước chính:
1. Khởi tạo run_id, backup code, tạo thư mục kết quả.
2. Đọc dữ liệu, tiền xử lý riêng cho tree‑based và neural models.
3. Chia train/val/test (stratified theo churn).
4. Huấn luyện Multitask DNN (với cân bằng dữ liệu) và chọn model tốt nhất.
5. Tạo giải thích (rule‑based) và thống kê churn cho Multitask DNN.
6. Stacking với base models (LGBM, RF, ET) và 3 meta‑learners (LR, DNN, RNN) cho từng task:
   - Cross‑validation trên base models để chọn fold tốt nhất.
   - Tạo meta‑features (probabilities) từ best models.
   - Huấn luyện meta‑learners, đánh giá trên test set.
   - Lưu kết quả vào các thư mục Scenario_1 (LR), Scenario_2 (DNN), Scenario_3 (RNN).
7. Tổng hợp kết quả (bao gồm CV của base models và stacking) và vẽ biểu đồ so sánh.
8. Đánh giá cost‑sensitive và profit‑based cho task churn.
9. Pipeline cảnh báo churn (Random Forest baseline).
10. So sánh giải thích (SHAP, LIME, rule) cho một số mẫu test.
"""

import os
import shutil
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from decision_tree_thresholds import extract_decision_tree_thresholds
from collections import Counter
from typing import Optional, List, Dict, Any
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from evaluations import plot_task_model_comparison

from data_ultils import (
    preprocess_tree_based, preprocess_neural,
    apply_balancing_all_tasks
)
from models import (
    build_lgbm_classifier, 
    build_rf_classifier, 
    build_et_classifier,
    build_xgb_classifier, 
    build_catboost_classifier,
    build_dnn_meta, 
    build_rnn_meta, 
    build_multitask_dnn,

)

from training import (
    cross_validate_base_models, train_meta_model,
    train_multitask_model
)

from explainers import (
explain_multitask_churn_shap,
    explain_multitask_churn_lime,
    )
from evaluations import (
    plot_label_distribution, plot_training_history,
    save_predictions_with_explanations, generate_churn_reason_statistics,
    plot_model_comparison, export_compare_results_to_csv,
    cost_sensitive_metrics, profit_based_metrics, aggregate_cv_results,
    run_churn_risk_alert_pipeline, plot_high_risk_reason_distribution,
    remove_top_right_spines, infer_reason, reason_to_features,
    plot_global_feature_importance, plot_jaccard_boxplot,
    plot_shap_summary, plot_waterfall
)
from decision_tree_thresholds import extract_decision_tree_thresholds
from explainers import (
    explain_multitask_churn_shap,
    explain_multitask_churn_lime,
    explain_multitask_churn_shap_global,
)

# Import các module tự viết
from configs import (
    RANDOM_STATE, OUTPUT_DIR, get_output_subdir, get_data_path,
    MULTITASK_DNN_DIRNAME, BASELINE_DIRNAME, CHURN_ALERT_DIRNAME,
    COST_FP, COST_FN, PROFIT_TP, PROFIT_TN,
    SHAP_NSAMPLES, EXPLANATION_TOP_K,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    BACKUP_DIRNAME,
    EXPLANATION_MAX_SAMPLES,
    N_FOLDS, N_REPEATS
)



# Đặt seed cho numpy và tensorflow
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def backup_code(run_id: str) -> None:
    """Sao lưu toàn bộ file .py vào thư mục backup."""
    
    backup_dir = os.path.join(OUTPUT_DIR, BACKUP_DIRNAME, run_id)
    os.makedirs(backup_dir, exist_ok=True)
    project_root = os.path.dirname(os.path.abspath(__file__))
    for root, _, files in os.walk(project_root):
        if os.path.abspath(root).startswith(os.path.abspath(os.path.join(OUTPUT_DIR, BACKUP_DIRNAME))):
            continue
        for f in files:
            if f.endswith(".py"):
                src = os.path.join(root, f)
                rel = os.path.relpath(root, project_root)
                dst_dir = os.path.join(backup_dir, rel)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, f))
    print(f"[BACKUP] Code saved to {backup_dir}")


def run_explanation_comparison(
    run_id: str,
    best_model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_true_churn_test: np.ndarray,
    y_churn_pred: np.ndarray,
    metadata_test: pd.DataFrame,
    df_full: pd.DataFrame,
    dataset_idx_test: Optional[np.ndarray] = None,
    background_size: int = None,
    max_samples: int = 30,
    random_state: int = 42,
    do_global_shap: bool = True,
    select_stratified_by_error: bool = True,
    thresholds_dict: dict = None,
    topk=EXPLANATION_TOP_K,
) -> Optional[pd.DataFrame]:
    """
    PHẦN 5 – SO SÁNH GIẢI THÍCH (RULE-BASED vs SHAP vs LIME)

    Tạo thư mục: OUTPUT/<run_id>/Explanation_Comparison
    Xuất:
      - xai_samples.csv
      - explanation_comparison.csv
      - similarity_summary.csv
      - xai_summary_by_error_group.csv
      - xai_summary_overall.csv
      - jaccard_heatmap.png
      - feature_frequency.png
      - (optional) shap_values_test.npy, expected_value.npy
      - (optional) global_feature_importance.png, jaccard_boxplot.png, shap_summary.png
      - (optional) waterfall_sample_<test_idx>_high.png / waterfall_sample_<test_idx>_low.png

    Return:
      df_comp (DataFrame) hoặc None nếu không chọn được mẫu.
    """
    np.random.seed(random_state)

    explain_comparison_dir = get_output_subdir(run_id, "Explanation_Comparison")
    os.makedirs(explain_comparison_dir, exist_ok=True)

    # 0) Tính p_churn cho toàn bộ test (để có p_churn + error_group)
    try:
        y_pred_full = best_model.predict(X_test, verbose=0)
        p_churn_test = np.asarray(y_pred_full[0]).reshape(-1)
    except Exception:
        p_churn_test = np.zeros(len(X_test), dtype=float)

    # 1) Background SHAP
    from configs import SHAP_BACKGROUND_SIZE
    bg_n = min(SHAP_BACKGROUND_SIZE, len(X_train))
    if bg_n <= 0:
        print("⚠️ X_train rỗng, bỏ qua PHẦN 5.")
        return None
    bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
    X_background = X_train[bg_idx]

    # 2) Feature names
    drop_cols = ["Exited", "CreditScoreClass", "HighBalanceFlag"]
    feature_names = df_full.drop(columns=drop_cols, errors="ignore").columns.tolist()
    print("[DEBUG] Feature names:", feature_names)

    # 3) Chọn mẫu để giải thích
    y_true = y_true_churn_test.astype(int)
    y_pred = y_churn_pred.astype(int)

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    tn = np.where((y_true == 0) & (y_pred == 0))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    def _err_group(t: int, p: int) -> str:
        if t == 1 and p == 1:
            return "TP"
        if t == 0 and p == 0:
            return "TN"
        if t == 0 and p == 1:
            return "FP"
        return "FN"

    if select_stratified_by_error:
        groups = [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]
        per_g = max_samples // 4
        remainder = max_samples - per_g * 4

        selected: List[int] = []
        for _, arr in groups:
            if len(arr) > 0 and per_g > 0:
                k = min(per_g, len(arr))
                selected.extend(np.random.choice(arr, size=k, replace=False).tolist())

        priority = ["FP", "FN", "TP", "TN"]
        gmap = {g: arr for g, arr in groups}
        for gname in priority:
            if remainder <= 0:
                break
            arr = gmap[gname]
            remaining_candidates = [x for x in arr.tolist() if x not in selected]
            if len(remaining_candidates) == 0:
                continue
            take = min(remainder, len(remaining_candidates))
            selected.extend(np.random.choice(remaining_candidates, size=take, replace=False).tolist())
            remainder -= take

        selected_idx = np.array(sorted(set(selected)), dtype=int)
        if len(selected_idx) == 0:
            print("⚠️ Không chọn được mẫu nào, bỏ qua.")
            return None
    else:
        churn_pred_indices = np.where(y_pred == 1)[0]
        if len(churn_pred_indices) == 0:
            print("⚠️ Không có mẫu churn nào trong test set, bỏ qua.")
            return None
        sample_size = min(max_samples, len(churn_pred_indices))
        selected_idx = np.random.choice(churn_pred_indices, size=sample_size, replace=False)

    # 4) Xuất xai_samples.csv
    xai_samples_rows: List[Dict[str, Any]] = []
    for t_idx in selected_idx:
        xai_samples_rows.append({
            "test_idx": int(t_idx),
            "dataset_idx": int(dataset_idx_test[t_idx]) if dataset_idx_test is not None else np.nan,
            "true_churn": int(y_true[t_idx]),
            "pred_churn": int(y_pred[t_idx]),
            "p_churn": float(p_churn_test[t_idx]),
            "error_group": _err_group(int(y_true[t_idx]), int(y_pred[t_idx])),
        })
    df_samples = pd.DataFrame(xai_samples_rows)
    df_samples_path = os.path.join(explain_comparison_dir, "xai_samples.csv")
    df_samples.to_csv(df_samples_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved xai samples to: {df_samples_path}")

    # 5) Chạy Rule/SHAP/LIME trên các mẫu đã chọn
    sample_instances = X_test[selected_idx]
    sample_metadata = metadata_test.iloc[selected_idx].reset_index(drop=True)

    comparison_rows: List[Dict[str, Any]] = []

    for i, inst in enumerate(sample_instances):
        inst_2d = inst.reshape(1, -1)
        row = sample_metadata.iloc[i]
        test_idx = int(selected_idx[i])
        true_ch = int(y_true[test_idx])
        pred_ch = int(y_pred[test_idx])
        p_ch = float(p_churn_test[test_idx])
        err_g = _err_group(true_ch, pred_ch)

        # Rule-based
        reason_str = infer_reason(row, thresholds_dict=thresholds_dict)
        rule_features = set(reason_to_features(reason_str))

        # SHAP
        try:
            shap_res = explain_multitask_churn_shap(best_model, X_background, inst_2d, feature_names)
            shap_top = shap_res.get("top_names", []) if shap_res else []
            shap_features = set(shap_top)
        except Exception as e:
            print(f"SHAP error on sample {i} (test_idx={test_idx}): {e}")
            shap_top = []
            shap_features = set()

        # LIME
        try:
            lime_res = explain_multitask_churn_lime(best_model, X_train, inst_2d, feature_names)
            lime_top = lime_res.get("top_names", []) if lime_res else []
            lime_features = set(lime_top)
        except Exception as e:
            print(f"LIME error on sample {i} (test_idx={test_idx}): {e}")
            lime_top = []
            lime_features = set()

        # Jaccard
        j_rule_shap = (
            len(rule_features & shap_features) / len(rule_features | shap_features)
        ) if (rule_features or shap_features) else 1.0
        j_rule_lime = (
            len(rule_features & lime_features) / len(rule_features | lime_features)
        ) if (rule_features or lime_features) else 1.0
        j_shap_lime = (
            len(shap_features & lime_features) / len(shap_features | lime_features)
        ) if (shap_features or lime_features) else 1.0

        # Coverage
        if len(rule_features) > 0:
            cov_shap5 = len(rule_features & shap_features) / len(rule_features)
            cov_lime5 = len(rule_features & lime_features) / len(rule_features)
        else:
            cov_shap5 = np.nan
            cov_lime5 = np.nan

        comparison_rows.append({
            "sample_id": int(i),
            "test_idx": int(test_idx),
            "dataset_idx": int(dataset_idx_test[test_idx]) if dataset_idx_test is not None else np.nan,
            "true_churn": int(true_ch),
            "pred_churn": int(pred_ch),
            "p_churn": float(p_ch),
            "error_group": err_g,
            "rule_reason": reason_str,
            "rule_features": ", ".join(sorted(rule_features)),
            "shap_top5": ", ".join(shap_top),
            "lime_top5": ", ".join(lime_top),
            "jaccard_rule_shap": float(j_rule_shap),
            "jaccard_rule_lime": float(j_rule_lime),
            "jaccard_shap_lime": float(j_shap_lime),
            "rule_coverage_shap5": float(cov_shap5) if cov_shap5 == cov_shap5 else np.nan,
            "rule_coverage_lime5": float(cov_lime5) if cov_lime5 == cov_lime5 else np.nan,
        })

    df_comp = pd.DataFrame(comparison_rows)
    comp_csv_path = os.path.join(explain_comparison_dir, "explanation_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved explanation comparison to: {comp_csv_path}")

    # 6) similarity_summary.csv + xai_summary_by_error_group.csv + xai_summary_overall.csv
    print("\n=== 📊 THỐNG KÊ SO SÁNH GIẢI THÍCH (OVERALL) ===")
    print(f"Jaccard (Rule vs SHAP) - Mean: {df_comp['jaccard_rule_shap'].mean():.3f}, Std: {df_comp['jaccard_rule_shap'].std():.3f}")
    print(f"Jaccard (Rule vs LIME) - Mean: {df_comp['jaccard_rule_lime'].mean():.3f}, Std: {df_comp['jaccard_rule_lime'].std():.3f}")
    print(f"Jaccard (SHAP vs LIME) - Mean: {df_comp['jaccard_shap_lime'].mean():.3f}, Std: {df_comp['jaccard_shap_lime'].std():.3f}")

    summary_stats = df_comp[["jaccard_rule_shap", "jaccard_rule_lime", "jaccard_shap_lime"]].describe()
    summary_stats.to_csv(os.path.join(explain_comparison_dir, "similarity_summary.csv"))

    grp = df_comp.groupby("error_group", dropna=False)
    df_by_group = grp.agg(
        n_samples=("test_idx", "count"),
        mean_p_churn=("p_churn", "mean"),
        mean_j_rule_shap=("jaccard_rule_shap", "mean"),
        std_j_rule_shap=("jaccard_rule_shap", "std"),
        mean_j_rule_lime=("jaccard_rule_lime", "mean"),
        std_j_rule_lime=("jaccard_rule_lime", "std"),
        mean_j_shap_lime=("jaccard_shap_lime", "mean"),
        std_j_shap_lime=("jaccard_shap_lime", "std"),
        mean_cov_shap5=("rule_coverage_shap5", "mean"),
        mean_cov_lime5=("rule_coverage_lime5", "mean"),
    ).reset_index()
    by_group_path = os.path.join(explain_comparison_dir, "xai_summary_by_error_group.csv")
    df_by_group.to_csv(by_group_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved xai summary by error group to: {by_group_path}")

    overall_row = {
        "n_samples": int(len(df_comp)),
        "mean_p_churn": float(df_comp["p_churn"].mean()),
        "mean_jaccard_rule_shap": float(df_comp["jaccard_rule_shap"].mean()),
        "std_jaccard_rule_shap": float(df_comp["jaccard_rule_shap"].std()),
        "mean_jaccard_rule_lime": float(df_comp["jaccard_rule_lime"].mean()),
        "std_jaccard_rule_lime": float(df_comp["jaccard_rule_lime"].std()),
        "mean_jaccard_shap_lime": float(df_comp["jaccard_shap_lime"].mean()),
        "std_jaccard_shap_lime": float(df_comp["jaccard_shap_lime"].std()),
        "mean_rule_coverage_shap5": float(df_comp["rule_coverage_shap5"].mean()),
        "mean_rule_coverage_lime5": float(df_comp["rule_coverage_lime5"].mean()),
        "selection_mode": "stratified_TP_TN_FP_FN" if select_stratified_by_error else "pred_churn_only",
        "topk": 5,
    }
    overall_path = os.path.join(explain_comparison_dir, "xai_summary_overall.csv")
    pd.DataFrame([overall_row]).to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved xai summary overall to: {overall_path}")

    # 7) Vẽ heatmap + feature frequency
    plt.figure(figsize=(10, 8))
    jaccard_matrix = df_comp[["jaccard_rule_shap", "jaccard_rule_lime", "jaccard_shap_lime"]].T
    sns.heatmap(jaccard_matrix, annot=False, cmap="YlOrRd", cbar_kws={"label": "Jaccard Similarity"})
    plt.xlabel("Sample index")
    plt.ylabel("Comparison")
    plt.yticks([0.5, 1.5, 2.5], ["Rule vs SHAP", "Rule vs LIME", "SHAP vs LIME"], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(explain_comparison_dir, "jaccard_heatmap.png"), dpi=300)
    plt.close()

    def count_features_from_set_list(feature_sets_list):
        counter = Counter()
        for s in feature_sets_list:
            counter.update(s)
        return counter

    rule_feature_sets = [set(f.split(", ") if f else []) for f in df_comp["rule_features"]]
    shap_feature_sets = [set(f.split(", ") if f else []) for f in df_comp["shap_top5"]]
    lime_feature_sets = [set(f.split(", ") if f else []) for f in df_comp["lime_top5"]]

    rule_counter = count_features_from_set_list(rule_feature_sets)
    shap_counter = count_features_from_set_list(shap_feature_sets)
    lime_counter = count_features_from_set_list(lime_feature_sets)

    feature_counts_df = pd.DataFrame({"Rule": rule_counter, "SHAP": shap_counter, "LIME": lime_counter}).fillna(0).astype(int)
    feature_counts_df["Total"] = feature_counts_df.sum(axis=1)
    feature_counts_df = feature_counts_df.sort_values("Total", ascending=False).drop("Total", axis=1)

    ax = feature_counts_df.plot(kind="bar", figsize=(10, 6), color=["#2ecc71", "#3498db", "#e74c3c"])
    remove_top_right_spines(ax)
    plt.xlabel("Feature")
    plt.ylabel("Frequency in Top-5")
    plt.legend(title="Method")
    plt.tight_layout()
    plt.savefig(os.path.join(explain_comparison_dir, "feature_frequency.png"), dpi=300)
    plt.close()

    # 8) Global SHAP + plots nâng cao (tùy chọn)
    if do_global_shap:
        try:
            shap_values, expected_value = explain_multitask_churn_shap_global(
                best_model, X_background, X_test, feature_names
            )
            np.save(os.path.join(explain_comparison_dir, "shap_values_test.npy"), shap_values)
            np.save(os.path.join(explain_comparison_dir, "expected_value.npy"), expected_value)

            rule_freq_global = df_comp["rule_features"].str.split(", ").explode().value_counts()
            lime_freq_global = df_comp["lime_top5"].str.split(", ").explode().value_counts()
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            plot_global_feature_importance(
                mean_abs_shap,
                rule_freq_global,
                lime_freq_global,
                feature_names,
                os.path.join(explain_comparison_dir, "global_feature_importance.png"),
            )

            plot_jaccard_boxplot(df_comp, os.path.join(explain_comparison_dir, "jaccard_boxplot.png"))

            plot_shap_summary(
                shap_values,
                X_test,
                feature_names,
                os.path.join(explain_comparison_dir, "shap_summary.png"),
            )

            high_test_idx = int(df_comp.loc[df_comp["jaccard_rule_shap"].idxmax(), "test_idx"])
            low_test_idx = int(df_comp.loc[df_comp["jaccard_rule_shap"].idxmin(), "test_idx"])

            plot_waterfall(
                shap_values,
                expected_value,
                high_test_idx,
                feature_names,
                X_test[high_test_idx],
                os.path.join(explain_comparison_dir, f"waterfall_sample_{high_test_idx}_high.png"),
            )
            plot_waterfall(
                shap_values,
                expected_value,
                low_test_idx,
                feature_names,
                X_test[low_test_idx],
                os.path.join(explain_comparison_dir, f"waterfall_sample_{low_test_idx}_low.png"),
            )
        except Exception as e:
            print(f"⚠️ Lỗi khi thực hiện SHAP toàn cục: {e}")
            import traceback
            traceback.print_exc()

    return df_comp


def main():
    # ==================================================================
    # 1. Khởi tạo run_id và thư mục kết quả
    # ==================================================================
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{now}"
    backup_code(run_id)
    result_folder = get_output_subdir(run_id)
    print(f"Run ID: {run_id}, results in {result_folder}")

    # ==================================================================
    # 2. Đọc và tiền xử lý dữ liệu
    # ==================================================================
    filepath = get_data_path()
    df_full = pd.read_csv(filepath)

    # Tree‑based pipeline (cho stacking và baseline)
    X_tree, y_dict_tree, df_tree = preprocess_tree_based(df_full)
    # Neural pipeline (cho Multitask DNN)
    X_neural, y_dict_neural, df_neural, scaler_neural = preprocess_neural(df_full)

    # Sử dụng neural pipeline cho Multitask DNN
    X = X_neural
    y_dict = y_dict_neural

    # ==================================================================
    # 3. Chia train/val/test (giữ nguyên index để đồng bộ metadata)
    # ==================================================================
    idx_all = np.arange(len(X))
    idx_train_full, idx_test = train_test_split(
        idx_all, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_dict['churn']
    )
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y_dict['churn'][idx_train_full]
    )

    # Dữ liệu neural cho Multitask DNN (đã scale)
    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]

    y_train_dict = {
        'churn': y_dict['churn'][idx_train],
        'score': y_dict['score'][idx_train],
        'balance': y_dict['balance'][idx_train]
    }
    y_val_dict = {
        'churn': y_dict['churn'][idx_val],
        'score': y_dict['score'][idx_val],
        'balance': y_dict['balance'][idx_val]
    }
    y_test_dict = {
        'churn': y_dict['churn'][idx_test],
        'score': y_dict['score'][idx_test],
        'balance': y_dict['balance'][idx_test]
    }

    # Dữ liệu tree cho stacking (chưa scale, chỉ encode)
    X_tree_train = X_tree[idx_train]
    X_tree_val = X_tree[idx_val]
    X_tree_test = X_tree[idx_test]

    y_tree_train_dict = {
        'churn': y_dict_tree['churn'][idx_train],
        'score': y_dict_tree['score'][idx_train],
        'balance': y_dict_tree['balance'][idx_train]
    }
    y_tree_test_dict = {
        'churn': y_dict_tree['churn'][idx_test],
        'score': y_dict_tree['score'][idx_test],
        'balance': y_dict_tree['balance'][idx_test]
    }

    # Metadata cho test (dùng cho giải thích)
    metadata_test = df_full.iloc[idx_test].reset_index(drop=True)

    # ==================================================================
    # 4. Cân bằng dữ liệu train cho Multitask DNN
    # ==================================================================
    X_train_bal, y_train_dict_bal = apply_balancing_all_tasks(X_train, y_train_dict)

    # Vẽ phân phối nhãn trước/sau cân bằng
    plot_label_distribution(y_dict, save_dir=result_folder, prefix="original")
    plot_label_distribution(y_train_dict_bal, save_dir=result_folder, prefix="balanced_train")

    # ==================================================================
    # 5. Huấn luyện Multitask DNN (grid search nhẹ)
    # ==================================================================
    multitask_dir = get_output_subdir(run_id, MULTITASK_DNN_DIRNAME)
    epoch_list = [DEFAULT_EPOCHS]          # luôn lấy từ configs.py
    batch_size_list = [DEFAULT_BATCH_SIZE]
    all_results = []
    model_paths = {}

    for epochs in epoch_list:
        for batch_size in batch_size_list:
            config_folder = get_output_subdir(run_id, MULTITASK_DNN_DIRNAME, f"ep{epochs}_bs{batch_size}")
            model = build_multitask_dnn(X.shape[1])
            model_path = os.path.join(config_folder, f"model_e{epochs}_b{batch_size}.keras")

            history, test_metrics = train_multitask_model(
                model=model,
                X_train=X_train_bal,
                y_train_dict=y_train_dict_bal,
                X_val=X_val,
                y_val_dict=y_val_dict,
                X_test=X_test,
                y_test_dict=y_test_dict,
                epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE,
                save_path=model_path,
                save_dir=config_folder,
                model_name="Multitask DNN"
            )
            plot_training_history(history, config_folder)
            model_paths[(epochs, batch_size)] = model_path
            all_results.extend(test_metrics)

    # Lưu kết quả grid search
    df_all_results = pd.DataFrame(all_results)
    df_all_results.to_csv(os.path.join(multitask_dir, "gridsearch_multitask_results.csv"), index=False)

    # Chọn model tốt nhất theo F1 + Recall cho churn
    df_churn = df_all_results[df_all_results["Task"] == "churn"].copy()
    df_churn["Score"] = df_churn["F1-score"] + df_churn["Recall"]
    best_row = df_churn.loc[df_churn["Score"].idxmax()]
    best_epochs = int(best_row["Epochs"])
    best_batch = int(best_row["Batch Size"])
    best_model_path = model_paths[(best_epochs, best_batch)]
    best_model = tf.keras.models.load_model(best_model_path)

    # Dự đoán trên test
    y_pred = best_model.predict(X_test, verbose=0)
    y_churn_pred = (y_pred[0] > 0.5).astype(int).flatten()
    y_score_pred = y_pred[1].argmax(axis=1)
    y_balance_pred = (y_pred[2] > 0.5).astype(int).flatten()

    # Lọc metrics của best model
    mlt_best_results = [
        r for r in all_results
        if r["Epochs"] == best_epochs and r["Batch Size"] == best_batch
    ]

    # ==================================================================
    # 6. Lấy thresholds từ Decision Tree (cho rule‑based explanations)
    # ==================================================================
    feature_cols = ['Balance', 'Tenure', 'Age']
    target_col = 'Exited'
    thresholds = extract_decision_tree_thresholds(
        csv_path=filepath,
        feature_cols=feature_cols,
        target_col=target_col,
        max_depth=3,
        random_state=RANDOM_STATE
    )
    thresholds_dict = {}
    for feature, thresh in thresholds:
        if feature == 'Balance':
            thresholds_dict['Balance_low'] = thresh
        elif feature == 'Tenure':
            thresholds_dict['Tenure_new'] = thresh
        elif feature == 'Age':
            thresholds_dict['Age_senior'] = thresh
    pd.DataFrame(list(thresholds_dict.items()), columns=["Rule", "Threshold"])\
      .to_csv(os.path.join(multitask_dir, "decision_tree_thresholds.csv"), index=False)

    # ==================================================================
    # 7. Tạo giải thích và thống kê churn cho Multitask DNN
    # ==================================================================
    explanation_path = os.path.join(multitask_dir, "explanation_results.csv")
    save_predictions_with_explanations(
        y_churn_pred, y_score_pred, y_balance_pred, metadata_test,
        explanation_path, thresholds_dict=thresholds_dict
    )
    generate_churn_reason_statistics(explanation_path, multitask_dir, thresholds_dict=thresholds_dict)

    # ==================================================================
    # 8. STACKING – Xây dựng 3 scenario cho mỗi task
    # ==================================================================
    baseline_dir = get_output_subdir(run_id, BASELINE_DIRNAME)
    all_baseline_results = []          # chứa cả kết quả CV và stacking
    base_builders = [
        build_lgbm_classifier,
        build_rf_classifier,
        build_et_classifier,
        build_xgb_classifier,
        build_catboost_classifier
    ]

    # Định nghĩa 3 meta‑learner scenarios
    stacking_scenarios = [
        {
            "name": "LR",
            "builder": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "type": "sklearn"
        },
        {
            "name": "DNN",
            "builder": None,  # sẽ khởi tạo sau
            "type": "dnn"
        },
        {
            "name": "RNN",
            "builder": None,  # sẽ khởi tạo sau
            "type": "rnn"
        }
    ]

    for task in ["churn", "score", "balance"]:
        print(f"\n=== Processing task: {task} ===")
        task_dir = os.path.join(baseline_dir, task)
        os.makedirs(task_dir, exist_ok=True)

        # Lấy dữ liệu tree
        X_tr = X_tree_train
        y_tr = y_tree_train_dict[task]
        X_te = X_tree_test
        y_te = y_tree_test_dict[task]

        # --- Cross‑validation cho base models ---
        # Lưu fold base model vào OUTPUT/{run_id}/Baselines/{task}/
        base_output_dir = os.path.join(baseline_dir, task)
        best_models, cv_records, _ = cross_validate_base_models(
            X=X_tr,
            y=y_tr,
            base_model_builders=base_builders,
            n_splits=N_FOLDS,
            random_state=RANDOM_STATE,
            task_name=task,
            use_smote=True,
            base_output_dir=base_output_dir
        )

        # Lưu kết quả CV (mean ± std)
        cv_summary = aggregate_cv_results(cv_records)
        cv_summary.to_csv(os.path.join(task_dir, "base_models_cv_summary.csv"), index=False)

        # Thêm vào all_baseline_results dưới dạng "Model_CV" với chuỗi mean±std
        for _, row in cv_summary.iterrows():
            all_baseline_results.append({
                "Task": task,
                "Model": f"{row['Model']}_CV",
                "Accuracy": f"{row['Accuracy_mean']:.3f}±{row['Accuracy_std']:.3f}",
                "F1-score": f"{row['F1_mean']:.3f}±{row['F1_std']:.3f}",
                "Precision": f"{row['Precision_mean']:.3f}±{row['Precision_std']:.3f}",
                "Recall": f"{row['Recall_mean']:.3f}±{row['Recall_std']:.3f}"
            })

        # --- Tạo meta‑features từ best models ---
        meta_train_list = []
        meta_test_list = []
        for name, model in best_models.items():
            # Luôn dùng DataFrame với đúng tên cột khi predict
            feature_cols = [col for col in df_tree.columns if col not in ['RowNumber', 'CustomerId', 'Surname', 'Exited', 'CreditScoreClass', 'HighBalanceFlag']]
            X_tr_df = pd.DataFrame(X_tr, columns=feature_cols)
            X_te_df = pd.DataFrame(X_te, columns=feature_cols)
            if hasattr(model, "predict_proba"):
                if task == "score":
                    proba_train = model.predict_proba(X_tr_df)
                    proba_test = model.predict_proba(X_te_df)
                else:
                    proba_train = model.predict_proba(X_tr_df)[:, 1].reshape(-1, 1)
                    proba_test = model.predict_proba(X_te_df)[:, 1].reshape(-1, 1)
            else:
                proba_train = model.predict(X_tr_df).reshape(-1, 1)
                proba_test = model.predict(X_te_df).reshape(-1, 1)

            meta_train_list.append(proba_train)
            meta_test_list.append(proba_test)

        if task == "score":
            # Ghép 3 bộ (3 classes) → 9 cột
            # Vẽ biểu đồ học cho DNN/RNN
            if scenario["type"] in ["dnn", "rnn"] and hasattr(meta_model, 'history'):
                plot_training_history(meta_model.history, scenario_dir)
            meta_train = np.hstack([p.reshape(p.shape[0], -1) for p in meta_train_list])
            meta_test = np.hstack([p.reshape(p.shape[0], -1) for p in meta_test_list])
        else:
            # Ghép 3 cột probabilities
            meta_train = np.hstack(meta_train_list)
            meta_test = np.hstack(meta_test_list)

        # --- Huấn luyện từng meta‑learner ---

        from sklearn.model_selection import RepeatedKFold
        stacking_cv_summary = []
        for scenario in stacking_scenarios:
            print(f"   CV stacking with meta: {scenario['name']}")
            scenario_dir = os.path.join(task_dir, f"Scenario_{scenario['name']}")
            os.makedirs(scenario_dir, exist_ok=True)

            kf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
            fold_metrics = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(meta_train)):
                X_tr_fold, X_val_fold = meta_train[train_idx], meta_train[val_idx]
                y_tr_fold, y_val_fold = y_tr[train_idx], y_tr[val_idx]
                # One-hot cho task score nếu cần
                if task == "score" and scenario["type"] != "sklearn":
                    y_tr_fold = to_categorical(y_tr_fold, num_classes=3)
                    y_val_fold = to_categorical(y_val_fold, num_classes=3)
                # Khởi tạo builder động cho DNN/RNN
                if scenario["type"] == "dnn":
                    if task == "score":
                        scenario["builder"] = lambda: build_dnn_meta(input_dim=meta_train.shape[1], n_classes=3)
                    else:
                        scenario["builder"] = lambda: build_dnn_meta(input_dim=meta_train.shape[1], n_classes=1)
                elif scenario["type"] == "rnn":
                    if task == "score":
                        scenario["builder"] = lambda: build_rnn_meta(input_dim=meta_train.shape[1], n_classes=3)
                    else:
                        scenario["builder"] = lambda: build_rnn_meta(input_dim=meta_train.shape[1], n_classes=1)

                # Tạo thư mục cho fold
                fold_dir = os.path.join(scenario_dir, f"fold_{fold_idx+1}")
                os.makedirs(fold_dir, exist_ok=True)

                if scenario["type"] in ["dnn", "rnn"]:
                    y_pred_val, y_proba_val, meta_model, metrics, history = train_meta_model(
                        meta_features_train=X_tr_fold,
                        y_train=y_tr_fold,
                        meta_features_test=X_val_fold,
                        y_test=y_val_fold,
                        meta_model_builder=scenario["builder"],
                        meta_model_type=scenario["type"],
                        task_name=task,
                        save_dir=fold_dir,
                        epochs=DEFAULT_EPOCHS,
                        batch_size=DEFAULT_BATCH_SIZE
                    )
                    # Lưu learning curve nếu có
                    if history is not None:
                        plot_training_history(history, fold_dir)
                    # Lưu model
                    meta_model.save(os.path.join(fold_dir, f"meta_model_{scenario['name']}.keras"))
                else:
                    y_pred_val, y_proba_val, meta_model, metrics, _ = train_meta_model(
                        meta_features_train=X_tr_fold,
                        y_train=y_tr_fold,
                        meta_features_test=X_val_fold,
                        y_test=y_val_fold,
                        meta_model_builder=scenario["builder"],
                        meta_model_type=scenario["type"],
                        task_name=task,
                        save_dir=fold_dir,
                        epochs=DEFAULT_EPOCHS,
                        batch_size=DEFAULT_BATCH_SIZE
                    )
                    # Lưu model
                    import joblib
                    joblib.dump(meta_model, os.path.join(fold_dir, f"meta_model_{scenario['name']}.pkl"))
                # Lưu prediction
                np.save(os.path.join(fold_dir, "y_pred_val.npy"), y_pred_val)
                # Lưu metrics
                import json
                with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                fold_metrics.append(metrics)

            # Tổng hợp mean/std các chỉ số stacking
            metrics_keys = ["Accuracy", "F1-score", "Precision", "Recall"]
            mean_std = {k+"_mean": np.mean([m[k] for m in fold_metrics]) for k in metrics_keys}
            mean_std.update({k+"_std": np.std([m[k] for m in fold_metrics]) for k in metrics_keys})
            mean_std["Task"] = task
            mean_std["Model"] = f"Stacking_{scenario['name']}"
            stacking_cv_summary.append(mean_std)

            # Train full meta-learner trên toàn bộ meta_train để dự đoán test (giữ nguyên logic cũ)
            y_train_meta = y_tr
            y_test_meta = y_te
            if task == "score" and scenario["type"] != "sklearn":
                y_train_meta = to_categorical(y_tr, num_classes=3)
                y_test_meta = to_categorical(y_te, num_classes=3)
            if scenario["type"] in ["dnn", "rnn"]:
                y_pred_test, y_proba_test, meta_model, metrics, history = train_meta_model(
                    meta_features_train=meta_train,
                    y_train=y_train_meta,
                    meta_features_test=meta_test,
                    y_test=y_test_meta,
                    meta_model_builder=scenario["builder"],
                    meta_model_type=scenario["type"],
                    task_name=task,
                    save_dir=scenario_dir,
                    epochs=DEFAULT_EPOCHS,
                    batch_size=DEFAULT_BATCH_SIZE
                )
                if history is not None:
                    plot_training_history(history, scenario_dir)
            else:
                y_pred_test, y_proba_test, meta_model, metrics, _ = train_meta_model(
                    meta_features_train=meta_train,
                    y_train=y_train_meta,
                    meta_features_test=meta_test,
                    y_test=y_test_meta,
                    meta_model_builder=scenario["builder"],
                    meta_model_type=scenario["type"],
                    task_name=task,
                    save_dir=scenario_dir,
                    epochs=DEFAULT_EPOCHS,
                    batch_size=DEFAULT_BATCH_SIZE
                )
            # Lưu metrics test như cũ
            result_row = {
                "Task": task,
                "Model": f"Stacking_{scenario['name']}",
                "Accuracy": round(metrics["Accuracy"], 3),
                "F1-score": round(metrics["F1-score"], 3),
                "Precision": round(metrics["Precision"], 3),
                "Recall": round(metrics["Recall"], 3)
            }
            all_baseline_results.append(result_row)

            # Lưu model và predictions
            if scenario["type"] == "sklearn":
                import joblib
                joblib.dump(meta_model, os.path.join(scenario_dir, f"meta_model_{scenario['name']}.pkl"))
            else:
                meta_model.save(os.path.join(scenario_dir, f"meta_model_{scenario['name']}.keras"))
            np.save(os.path.join(scenario_dir, "y_pred_test.npy"), y_pred_test)

            # Nếu task là churn, lưu lại để dùng cho cost/profit sau
            if task == "churn" and scenario["name"] == "LR":
                stacking_churn_pred = y_pred_test
                stacking_churn_proba = y_proba_test

        # Lưu file tổng hợp mean/std stacking cho task này
        pd.DataFrame(stacking_cv_summary).to_csv(os.path.join(task_dir, "stacking_cv_summary.csv"), index=False)
        # Vẽ biểu đồ so sánh các chỉ số hiệu năng cho từng task
        
        plot_task_model_comparison(task_dir, task)

    # ==================================================================
    # 9. Tổng hợp kết quả và vẽ biểu đồ so sánh
    # ==================================================================
    compare_csv_path = os.path.join(result_folder, "compare_models_summary.csv")
    df_compare = export_compare_results_to_csv(
        mlt_results=mlt_best_results,
        baseline_results=all_baseline_results,
        output_path=compare_csv_path
    )

    # Vẽ biểu đồ cho các model có giá trị số (bỏ qua dạng "mean±std")
    numeric_results = [r for r in all_baseline_results if isinstance(r["Accuracy"], (int, float))]
    if numeric_results:
        df_num = pd.DataFrame(numeric_results)
        for metric in ["Accuracy", "F1-score", "Precision", "Recall"]:
            plot_model_comparison(df_num, metric=metric, save_dir=result_folder)

    # ==================================================================
    # 10. Cost‑sensitive và Profit‑based evaluation cho churn
    # ==================================================================
    y_true_churn = y_tree_test_dict['churn']
    # Dùng stacking LR (hoặc có thể dùng Random Forest)
    # Ở đây dùng stacking LR đã lưu ở trên
    cost_res = cost_sensitive_metrics(y_true_churn, stacking_churn_pred, cost_fp=COST_FP, cost_fn=COST_FN)
    profit_res = profit_based_metrics(y_true_churn, stacking_churn_pred,
                                       profit_tp=PROFIT_TP, profit_tn=PROFIT_TN,
                                       cost_fp=COST_FP, cost_fn=COST_FN)
    pd.DataFrame([cost_res]).to_csv(os.path.join(result_folder, "cost_sensitive_metrics_stacking_LR.csv"), index=False)
    pd.DataFrame([profit_res]).to_csv(os.path.join(result_folder, "profit_based_metrics_stacking_LR.csv"), index=False)

    # ==================================================================
    # 11. Pipeline cảnh báo churn (Random Forest)
    # ==================================================================
    run_churn_risk_alert_pipeline(filepath, result_folder)
    plot_high_risk_reason_distribution(
        csv_path=os.path.join(result_folder, CHURN_ALERT_DIRNAME, "high_risk_customers_with_reasons.csv"),
        output_path=os.path.join(result_folder, CHURN_ALERT_DIRNAME, "high_risk_reason_barplot.png")
    )

    # ==================================================================
    # 12. So sánh giải thích (SHAP, LIME, rule) – optional
    # ==================================================================
    
    run_explanation_comparison(
        run_id=run_id,
        best_model=best_model,
        X_train=X_train_bal,
        X_test=X_test,
        y_true_churn_test=y_test_dict['churn'],
        y_churn_pred=y_churn_pred,
        metadata_test=metadata_test,
        df_full=df_full,
        dataset_idx_test=idx_test,
        background_size=SHAP_NSAMPLES,
        max_samples=EXPLANATION_MAX_SAMPLES,
        random_state=RANDOM_STATE,
        do_global_shap=True,
        select_stratified_by_error=True,
        thresholds_dict=thresholds_dict,
        topk=EXPLANATION_TOP_K
    )

    print(f"\n✅ Pipeline hoàn tất. Kết quả lưu tại: {result_folder}")


if __name__ == "__main__":
    main()