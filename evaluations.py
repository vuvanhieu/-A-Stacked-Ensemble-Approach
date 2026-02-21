# File: evaluations.py
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import shap
import lime


from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.preprocessing import label_binarize

# Import các hằng số từ configs
from configs import (
    LOW_CREDIT_THRESHOLD, HIGH_CREDIT_THRESHOLD,
    LOW_BALANCE_THRESHOLD, HIGH_BALANCE_THRESHOLD,
    NEW_CUSTOMER_TENURE, SENIOR_AGE_THRESHOLD,
    INACTIVE_MEMBER_FLAG, LOW_PRODUCT_COUNT, HIGH_PRODUCT_COUNT, NO_CREDIT_CARD_FLAG,
    BASE_FONT_SIZE, TITLE_FONT_SIZE, LABEL_FONT_SIZE,
    TICK_FONT_SIZE, LEGEND_FONT_SIZE,
    COST_FP, COST_FN, PROFIT_TP, PROFIT_TN
)

# ==== Global Plot Font Config ====
plt.rcParams['font.size'] = BASE_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE

sns.set_context("notebook", rc={
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})

# ============================================================
#  PLOT TASK-WISE MODEL COMPARISON (MEAN SCORES)
# ============================================================
def plot_task_model_comparison(task_dir, task_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Đọc kết quả base models và stacking
    base_path = os.path.join(task_dir, "base_models_cv_summary.csv")
    stack_path = os.path.join(task_dir, "stacking_cv_summary.csv")
    if not (os.path.exists(base_path) and os.path.exists(stack_path)):
        print(f"Missing summary files for {task_name}")
        return

    df_base = pd.read_csv(base_path)
    df_stack = pd.read_csv(stack_path)

    # Đổi tên cột cho đồng nhất nếu cần
    if "F1_mean" in df_base.columns:
        df_base = df_base.rename(columns={
            "F1_mean": "F1-score_mean",
            "F1_std": "F1-score_std"
        })

    # Chỉ lấy các cột mean đã đổi tên
    cols = ["Model", "Accuracy_mean", "Precision_mean", "Recall_mean", "F1-score_mean"]
    df_plot = pd.concat([df_base[cols], df_stack[cols]], ignore_index=True)

    # Vẽ từng chỉ số
    for metric in ["Accuracy_mean", "Precision_mean", "Recall_mean", "F1-score_mean"]:
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=df_plot, x="Model", y=metric, palette="Set2")

        # Bỏ khung trên và phải
        remove_top_right_spines(ax)

        # ✅ Xoay nghiêng nhãn trục X
        ax.tick_params(axis="x", labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")  # canh phải cho đẹp, tránh đè chữ

        # Gắn nhãn lên cột
        for c in ax.containers:
            ax.bar_label(
                c, fmt="%.3f", padding=3, fontsize=10,
                label_type="edge", rotation=90, color="black"
            )

        plt.ylabel(metric)

        # Legend (thường barplot này không có legend, nhưng giữ lại cho an toàn)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_rotation(30)

        plt.tight_layout()
        out_path = os.path.join(task_dir, f"compare_{metric}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
        
        
def remove_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def get_display_labels(task_name, labels):
    """Chuyển nhãn số sang nhãn cụ thể cho từng task."""
    name = str(task_name).lower()
    if name == "churn":
        mapping = {0: "No churn", 1: "Churn"}
    elif "score" in name:
        mapping = {0: "Low score", 1: "Medium score", 2: "High score"}
    elif "balance" in name:
        mapping = {0: "Low balance", 1: "High balance"}
    else:
        mapping = {}
    display_labels = []
    for lb in labels:
        try:
            lb_int = int(lb)
        except:
            lb_int = lb
        display_labels.append(mapping.get(lb_int, str(lb)))
    return display_labels

# ============================================================
#  PLOT CONFUSION MATRIX + ROC + PR + AUC
# ============================================================
def plot_all(y_true, y_pred, y_proba, task_name, save_dir, model_name=""):
    """Vẽ confusion matrix (chuẩn và chuẩn hóa), ROC, PR, lưu AUC."""
    os.makedirs(save_dir, exist_ok=True)
    postfix = f"_{model_name}" if model_name else ""

    # One-hot → label indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Normalize proba to shape Nx2 for binary
    if y_proba.ndim == 1:
        y_proba = np.vstack([1 - y_proba, y_proba]).T
    elif y_proba.ndim == 2 and y_proba.shape[1] == 1:
        y_proba = np.hstack([1 - y_proba, y_proba])

    labels = np.unique(np.concatenate([y_true, y_pred]))
    display_labels = get_display_labels(task_name, labels)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    ax = plt.gca()
    remove_top_right_spines(ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix{postfix}.png"), dpi=300)
    plt.close()

    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    df_cm_norm = pd.DataFrame(cm_norm, index=display_labels, columns=display_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="YlGnBu")
    ax = plt.gca()
    remove_top_right_spines(ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{task_name}_confusion_matrix_normalized{postfix}.png"), dpi=300)
    plt.close()

    if len(np.unique(y_true)) < 2:
        print(f"Skip ROC/PR/AUC for '{task_name}' because there is only one class.")
        return

    # ROC Curve
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    try:
        if y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[1], lw=2, label=f"AUC = {roc_auc:.2f}")
        else:
            y_true_bin = label_binarize(y_true, classes=labels)
            for i, lb in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                class_name = display_labels[i]
                plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{class_name} (AUC={auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        ax = plt.gca()
        remove_top_right_spines(ax)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_roc_curve{postfix}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"ROC Error for task '{task_name}': {e}")

    # Precision–Recall Curve
    plt.figure(figsize=(7, 6))
    colors2 = plt.cm.Dark2(np.linspace(0, 1, len(labels)))
    try:
        if y_proba.shape[1] == 2:
            prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
            pr_auc = auc(rec, prec)
            plt.plot(rec, prec, color=colors2[1], lw=2, label=f"AUC = {pr_auc:.2f}")
        else:
            y_bin = label_binarize(y_true, classes=labels)
            for i, lb in enumerate(labels):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                pr_auc = auc(rec, prec)
                class_name = display_labels[i]
                plt.plot(rec, prec, color=colors2[i], lw=2, label=f"{class_name} (AUC={pr_auc:.2f})")
        ax = plt.gca()
        remove_top_right_spines(ax)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{task_name}_pr_curve{postfix}.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"PR Error for task '{task_name}': {e}")

    # AUC Score File
    try:
        if y_proba.shape[1] == 2:
            auc_score_val = roc_auc_score(y_true, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=labels)
            auc_score_val = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
        with open(os.path.join(save_dir, f"{task_name}_AUC{postfix}.txt"), "w") as f:
            f.write(f"AUC ({task_name} - {model_name}): {auc_score_val:.4f}\n")
    except Exception as e:
        print(f"AUC file write error for task '{task_name}': {e}")

# ============================================================
#  PLOT LABEL DISTRIBUTION
# ============================================================
def plot_label_distribution(y_dict, save_dir=".", prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    for task_name, y in y_dict.items():
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        label_counts = pd.Series(y).value_counts().sort_index()
        percentages = label_counts / len(y) * 100
        raw_labels = label_counts.index.values
        display_labels = get_display_labels(task_name, raw_labels)
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=raw_labels, y=label_counts.values, palette="Spectral")
        remove_top_right_spines(ax)
        ax.set_xticks(range(len(display_labels)))
        ax.set_xticklabels(display_labels)
        for i, count in enumerate(label_counts.values):
            ax.text(i, count + 0.5, f"{count} ({percentages.iloc[i]:.1f}%)",
                    ha='center', va='bottom', fontsize=TICK_FONT_SIZE)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        tag = f"{prefix}_{task_name}" if prefix else task_name
        out_path = os.path.join(save_dir, f"{tag}_distribution.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved label distribution: {out_path}")

# ============================================================
#  HIGH-RISK CHURN REASONS (RULE-BASED)
# ============================================================
def infer_reason(row, thresholds_dict=None):
    reasons = []
    credit_score = row.get("CreditScore", np.nan)
    if thresholds_dict and "CreditScore_low" in thresholds_dict:
        if credit_score < thresholds_dict["CreditScore_low"]:
            reasons.append("Low credit score")
    elif thresholds_dict and "CreditScore_high" in thresholds_dict:
        if credit_score > thresholds_dict["CreditScore_high"]:
            reasons.append("High credit score")
    else:
        if credit_score < LOW_CREDIT_THRESHOLD:
            reasons.append("Low credit score")
        elif credit_score > HIGH_CREDIT_THRESHOLD:
            reasons.append("High credit score")

    balance = row.get("Balance", np.nan)
    if thresholds_dict and "Balance_high" in thresholds_dict:
        if balance > thresholds_dict["Balance_high"]:
            reasons.append("High balance")
    elif thresholds_dict and "Balance_low" in thresholds_dict:
        if balance < thresholds_dict["Balance_low"]:
            reasons.append("Very low balance")
    else:
        if balance > HIGH_BALANCE_THRESHOLD:
            reasons.append("High balance")
        elif balance < LOW_BALANCE_THRESHOLD:
            reasons.append("Very low balance")

    tenure = row.get("Tenure", np.nan)
    if thresholds_dict and "Tenure_new" in thresholds_dict:
        if tenure <= thresholds_dict["Tenure_new"]:
            reasons.append("New customer")
    else:
        if tenure <= NEW_CUSTOMER_TENURE:
            reasons.append("New customer")

    age = row.get("Age", np.nan)
    if thresholds_dict and "Age_senior" in thresholds_dict:
        if age >= thresholds_dict["Age_senior"]:
            reasons.append("Older customer")
    else:
        if age >= SENIOR_AGE_THRESHOLD:
            reasons.append("Older customer")

    if "IsActiveMember" in row:
        try:
            if int(row["IsActiveMember"]) == int(INACTIVE_MEMBER_FLAG):
                reasons.append("Inactive member")
        except:
            pass

    if "NumOfProducts" in row:
        try:
            nprod = int(row["NumOfProducts"])
            if thresholds_dict and "NumOfProducts_low" in thresholds_dict:
                if nprod <= int(thresholds_dict["NumOfProducts_low"]):
                    reasons.append("Low product engagement")
            elif nprod <= int(LOW_PRODUCT_COUNT):
                reasons.append("Low product engagement")
            if thresholds_dict and "NumOfProducts_high" in thresholds_dict:
                if nprod >= int(thresholds_dict["NumOfProducts_high"]):
                    reasons.append("Many products")
            elif nprod >= int(HIGH_PRODUCT_COUNT):
                reasons.append("Many products")
        except:
            pass

    if "HasCrCard" in row:
        try:
            if int(row["HasCrCard"]) == int(NO_CREDIT_CARD_FLAG):
                reasons.append("No credit card")
        except:
            pass

    return ", ".join(reasons)

def generate_high_risk_recommendations(csv_in, csv_out):
    df = pd.read_csv(csv_in)
    high_df = df[df["Risk_Level"] == "High"].copy()
    high_df["Churn_Reason_Suggestion"] = high_df.apply(infer_reason, axis=1)
    high_df.to_csv(csv_out, index=False)
    print(f"Saved high-risk recommendations: {csv_out}")

def plot_high_risk_reason_distribution(csv_path, output_path):
    df = pd.read_csv(csv_path)
    if "Churn_Reason_Suggestion" not in df.columns:
        print("Missing Churn_Reason_Suggestion column.")
        return
    reason_list = df["Churn_Reason_Suggestion"].dropna().str.split(", ")
    counter = Counter()
    for reasons in reason_list:
        counter.update(reasons)
    reason_df = pd.DataFrame(counter.items(), columns=["Reason", "Count"])
    reason_df = reason_df.sort_values("Count", ascending=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=reason_df, x="Count", y="Reason", palette="viridis")
    remove_top_right_spines(ax)
    plt.xlabel("Frequency")
    plt.ylabel("Churn Reason")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved high-risk churn reason plot: {output_path}")

# ============================================================
#  TRAINING HISTORY
# ============================================================
def plot_training_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Accuracy
    plt.figure(figsize=(10, 5))
    accuracy_keys = [k for k in history.history.keys() if 'accuracy' in k]
    legend_labels = []
    for key in accuracy_keys:
        plt.plot(history.history[key], lw=2)
        if key.startswith('val_'):
            task = key.replace('val_', '').replace('_accuracy', '')
            legend_labels.append(f"Validation {task.capitalize()}")
        else:
            task = key.replace('_accuracy', '')
            legend_labels.append(f"Training {task.capitalize()}")
    ax = plt.gca()
    remove_top_right_spines(ax)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "training_validation_accuracy.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()
    # Loss
    plt.figure(figsize=(10, 5))
    loss_keys = [k for k in history.history.keys() if 'loss' in k]
    legend_labels = []
    for key in loss_keys:
        plt.plot(history.history[key], lw=2)
        if key.startswith('val_'):
            task = key.replace('val_', '').replace('_loss', '')
            legend_labels.append(f"Validation {task.capitalize()}")
        else:
            task = key.replace('_loss', '')
            legend_labels.append(f"Training {task.capitalize()}")
    ax = plt.gca()
    remove_top_right_spines(ax)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "training_validation_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    return acc_path, loss_path

# ============================================================
#  SIMPLE EXPLANATIONS (MULTITASK PREDICTIONS)
# ============================================================
def generate_simple_explanations(y_churn, y_score, y_balance, metadata_df, max_samples=None, thresholds_dict=None):
    out = []
    for i in range(len(y_churn)):
        if y_churn[i] == 1:
            rs = []
            if y_score[i] == 0:
                rs.append("low credit score")
            elif y_score[i] == 2:
                rs.append("high credit score")
            rs.append("high balance" if y_balance[i] == 1 else "low balance")
            if thresholds_dict is not None:
                reason = infer_reason(metadata_df.iloc[i], thresholds_dict=thresholds_dict)
                rs.append(reason)
            text = f"Churn = 1 -> {', '.join(rs)} | Age: {metadata_df.iloc[i]['Age']}, Country: {metadata_df.iloc[i]['Geography']}"
        else:
            text = "No churn"
        out.append(text)
        if max_samples and len(out) >= max_samples:
            break
    return out

def save_predictions_with_explanations(y_churn, y_score, y_balance, metadata_df, output_path, thresholds_dict=None):
    expl = generate_simple_explanations(y_churn, y_score, y_balance, metadata_df, thresholds_dict=thresholds_dict)
    df_exp = metadata_df.copy()
    df_exp["Pred_Exited"] = y_churn
    df_exp["Pred_ScoreClass"] = y_score
    df_exp["Pred_HighBalance"] = y_balance
    df_exp["Explanation"] = expl
    df_exp.to_csv(output_path, index=False)
    print(f"Saved explanations: {output_path}")

# ============================================================
#  MODEL COMPARISON PLOT
# ============================================================
def plot_model_comparison(df_compare, metric='F1-score', save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_compare, x="Task", y=metric, hue="Model", palette="Accent")
    remove_top_right_spines(ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3, fontsize=TICK_FONT_SIZE, label_type="edge", rotation=90, color="black")
    plt.ylabel(metric)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"compare_{metric.lower()}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

# ============================================================
#  CHURN REASON STATISTICS FROM EXPLANATION CSV
# ============================================================
def generate_churn_reason_statistics(explanation_csv_path, output_dir=".", thresholds_dict=None):
    if not os.path.exists(explanation_csv_path):
        print(f"Explanation file not found: {explanation_csv_path}")
        return
    df = pd.read_csv(explanation_csv_path)
    if "Pred_Exited" not in df.columns or "Explanation" not in df.columns:
        print("Missing 'Pred_Exited' or 'Explanation' column.")
        return
    df_churn = df[df["Pred_Exited"] == 1].copy()
    if df_churn.empty:
        print("No churn samples found in explanation file.")
        return
    reason_counter = Counter()
    for text in df_churn["Explanation"].dropna():
        if "->" in text:
            reason_part = text.split("->", 1)[1]
        else:
            reason_part = text
        if "|" in reason_part:
            reason_part = reason_part.split("|", 1)[0]
        reasons = [r.strip() for r in reason_part.split(",") if r.strip()]
        reason_counter.update(reasons)
    if not reason_counter:
        print("No reasons extracted from explanations.")
        return
    reason_df = pd.DataFrame(reason_counter.items(), columns=["Reason", "Count"])
    reason_df = reason_df.sort_values("Count", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    csv_out = os.path.join(output_dir, "churn_reason_statistics.csv")
    fig_out = os.path.join(output_dir, "churn_reason_barplot.png")
    reason_df.to_csv(csv_out, index=False)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=reason_df, x="Count", y="Reason", palette="magma")
    remove_top_right_spines(ax)
    plt.xlabel("Count")
    plt.ylabel("Reason")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=300)
    plt.close()
    print(f"Saved churn reason statistics: {csv_out}")
    print(f"Saved churn reason barplot: {fig_out}")

# ============================================================
#  CHURN RISK ALERT PIPELINE (BASELINE RF)
# ============================================================
def classify_alert(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

def run_churn_risk_alert_pipeline(filepath, result_folder):
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        return
    df = pd.read_csv(filepath)
    df_enc = df.copy()
    if df_enc["Geography"].dtype == object:
        df_enc["Geography"] = pd.factorize(df_enc["Geography"])[0]
    if df_enc["Gender"].dtype == object:
        df_enc["Gender"] = pd.factorize(df_enc["Gender"])[0]
    df_enc["CreditScoreClass"] = pd.cut(
        df_enc["CreditScore"], bins=[0, 580, 700, 850], labels=[0, 1, 2]
    ).astype(int)
    df_enc["HighBalanceFlag"] = (df_enc["Balance"] > HIGH_BALANCE_THRESHOLD).astype(int)
    X = df_enc.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)
    y = df_enc["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    churn_risk_score = np.round(probs * 100).astype(int)
    alert_levels = [classify_alert(score) for score in churn_risk_score]
    output_dir = os.path.join(result_folder, "churn_risk_alert")
    os.makedirs(output_dir, exist_ok=True)
    alert_df = X_test.copy()
    alert_df["True_Exited"] = y_test.values
    alert_df["Churn_Prob"] = probs
    alert_df["Churn_Risk_Score"] = churn_risk_score
    alert_df["Risk_Level"] = alert_levels
    alert_log_path = os.path.join(output_dir, "churn_risk_alert_log.csv")
    alert_df.to_csv(alert_log_path, index=False)
    print(f"Alert log saved to {alert_log_path}")
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    report_path = os.path.join(output_dir, "churn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=alert_df, x="Risk_Level", order=["Low", "Medium", "High"], palette="Set2")
    remove_top_right_spines(ax)
    plt.ylabel("Number of Customers")
    plt.xlabel("Risk Level")
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, "churn_risk_level_distribution.png")
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved risk level distribution plot: {barplot_path}")
    high_risk_path = os.path.join(output_dir, "high_risk_customers_with_reasons.csv")
    generate_high_risk_recommendations(alert_log_path, high_risk_path)

def export_compare_results_to_csv(mlt_results, baseline_results, output_path="compare_results.csv"):
    all_rows = []
    for r in mlt_results:
        all_rows.append({
            "Task": r["Task"],
            "Model": "Multitask DNN",
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1-score": r["F1-score"]
        })
    for r in baseline_results:
        all_rows.append({
            "Task": r["Task"],
            "Model": r["Model"],
            "Accuracy": r["Accuracy"] if isinstance(r["Accuracy"], (int, float)) else r["Accuracy"],
            "Precision": r["Precision"] if isinstance(r["Precision"], (int, float)) else r["Precision"],
            "Recall": r["Recall"] if isinstance(r["Recall"], (int, float)) else r["Recall"],
            "F1-score": r["F1-score"] if isinstance(r["F1-score"], (int, float)) else r["F1-score"]
        })
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved comparison CSV to: {output_path}")
    return df

def reason_to_features(reason_str):
    if not isinstance(reason_str, str) or not reason_str.strip():
        return set()
    mapping = {
        'Low credit score': 'CreditScore',
        'High credit score': 'CreditScore',
        'High balance': 'Balance',
        'Very low balance': 'Balance',
        'New customer': 'Tenure',
        'Older customer': 'Age',
        'Inactive member': 'IsActiveMember',
        'Low product engagement': 'NumOfProducts',
        'Many products': 'NumOfProducts',
        'No credit card': 'HasCrCard',
    }
    features = set()
    parts = reason_str.split(',')
    for p in parts:
        p = p.strip()
        if p in mapping:
            features.add(mapping[p])
    return features

# ============================================================
#  COST-SENSITIVE & PROFIT-BASED EVALUATION
# ============================================================
def cost_sensitive_metrics(y_true, y_pred, cost_fp=COST_FP, cost_fn=COST_FN):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    avg_cost = total_cost / len(y_true)
    return {
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "fp": fp,
        "fn": fn
    }

def profit_based_metrics(y_true, y_pred, profit_tp=PROFIT_TP, profit_tn=PROFIT_TN, cost_fp=COST_FP, cost_fn=COST_FN):
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_profit = tp * profit_tp + tn * profit_tn - fp * cost_fp - fn * cost_fn
    avg_profit = total_profit / len(y_true)
    return {
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "profit_tp": profit_tp,
        "profit_tn": profit_tn,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn
    }

# ============================================================
#  HÀM TÍNH TRUNG BÌNH VÀ ĐỘ LỆCH CHO KẾT QUẢ CV
# ============================================================
def aggregate_cv_results(cv_records):
    """cv_records: list of dict, mỗi dict chứa 'Model', 'Fold', các metric"""
    df = pd.DataFrame(cv_records)
    summary = df.groupby("Model").agg(
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_std=("Accuracy", "std"),
        Precision_mean=("Precision", "mean"),
        Precision_std=("Precision", "std"),
        Recall_mean=("Recall", "mean"),
        Recall_std=("Recall", "std"),
        F1_mean=("F1-score", "mean"),
        F1_std=("F1-score", "std")
    ).reset_index()
    return summary



# ============================================================
#  CÁC HÀM VẼ CHO SO SÁNH GIẢI THÍCH (SHAP, LIME, RULE)
# ============================================================

def plot_global_feature_importance(mean_shap, rule_freq, lime_freq, feature_names, save_path):
    """Vẽ bar chart so sánh tầm quan trọng feature toàn cục."""
    # Fill missing values for all features
    feature_names = list(feature_names)
    mean_shap = np.asarray(mean_shap).flatten()
    # Nếu mean_shap không đủ độ dài, fill 0
    if len(mean_shap) < len(feature_names):
        mean_shap = np.concatenate([mean_shap, np.zeros(len(feature_names) - len(mean_shap))])
    elif len(mean_shap) > len(feature_names):
        mean_shap = mean_shap[:len(feature_names)]

    rule_freq = pd.Series(rule_freq).reindex(feature_names).fillna(0)
    lime_freq = pd.Series(lime_freq).reindex(feature_names).fillna(0)

    try:
        df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP (|value|)': mean_shap / mean_shap.max() if mean_shap.max() != 0 else mean_shap,
            'Rule frequency': rule_freq / rule_freq.max() if rule_freq.max() != 0 else rule_freq,
            'LIME frequency': lime_freq / lime_freq.max() if lime_freq.max() != 0 else lime_freq
        })
        df = df.melt(id_vars='Feature', var_name='Method', value_name='Normalized Importance')
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Feature', y='Normalized Importance', hue='Method')
        remove_top_right_spines(ax)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[DEBUG] plot_global_feature_importance error: {e}")
        print(f"mean_shap: {mean_shap.shape}, rule_freq: {rule_freq.shape}, lime_freq: {lime_freq.shape}, feature_names: {len(feature_names)}")


def plot_jaccard_boxplot(df_comp, save_path):
    """Vẽ boxplot cho 3 cặp Jaccard similarity."""
    plt.figure(figsize=(8, 6))
    data = df_comp[['jaccard_rule_shap', 'jaccard_rule_lime', 'jaccard_shap_lime']]
    ax = sns.boxplot(data=data)
    remove_top_right_spines(ax)
    ax.set_xticklabels(['Rule vs SHAP', 'Rule vs LIME', 'SHAP vs LIME'])
    ax.set_ylabel('Jaccard Similarity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_shap_summary(shap_values, X, feature_names, save_path):
    """Vẽ SHAP summary plot (beeswarm)."""
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_waterfall(shap_values, expected_value, instance_idx, feature_names, instance_data, save_path):
    """Vẽ waterfall plot cho một mẫu cụ thể."""
    values = shap_values[instance_idx]
    if hasattr(values, 'shape') and len(values.shape) > 1:
        values = values.flatten()
    shap.plots.waterfall(shap.Explanation(values=values,
                                          base_values=expected_value,
                                          data=instance_data,
                                          feature_names=feature_names),
                         max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
