import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def _normalize_lime_feature_name(lime_rule: str, feature_names: list) -> str:
    import re
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(lime_rule))
    for t in tokens:
        if t in feature_names:
            return t
    return tokens[0] if tokens else str(lime_rule)

def explain_multitask_churn_shap(model, X_background, X_instance, feature_names):
    def model_churn(x):
        return model.predict(x, verbose=0)[0]
    explainer = shap.KernelExplainer(model_churn, X_background)
    shap_values = explainer.shap_values(X_instance, nsamples=200)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    shap_values = shap_values.flatten()
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
    k = 5
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[-k:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    return {
        'shap_values': shap_values,
        'expected_value': expected_value,
        'top_indices': top_idx,
        'top_names': top_names,
        'top_values': shap_values[top_idx]
    }

def explain_multitask_churn_lime(model, X_train, X_instance, feature_names, class_names=['No Churn', 'Churn']):
    def predict_proba(x):
        proba_1 = model.predict(x, verbose=0)[0]
        proba_0 = 1 - proba_1
        return np.hstack([proba_0, proba_1])
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, feature_names=feature_names, class_names=class_names,
        mode='classification', discretize_continuous=False, random_state=42
    )
    exp = explainer.explain_instance(X_instance[0], predict_proba, num_features=5, top_labels=1)
    label = exp.top_labels[0]
    feature_weights = exp.as_list(label=label)
    top_features = []
    top_weights = []
    for f, w in feature_weights[:5]:
        fname = _normalize_lime_feature_name(f, feature_names)
        top_features.append(fname)
        top_weights.append(w)
    return {
        'explanation': exp,
        'top_names': top_features,
        'top_weights': top_weights,
        'label': label
    }

def get_top_feature_set(explanation_result, method):
    if method == 'shap':
        return set(explanation_result['top_names'])
    elif method == 'lime':
        return set(explanation_result['top_names'])
    else:
        return set()

def explain_multitask_churn_shap_global(model, X_background, X_explain, feature_names):
    def model_churn(x):
        return model.predict(x, verbose=0)[0]
    explainer = shap.KernelExplainer(model_churn, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=200)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) == 2 else expected_value[0]
    return shap_values, expected_value



# ============================================================
#  CÁC HÀM VẼ CHO SO SÁNH GIẢI THÍCH (SHAP, LIME, RULE)
# ============================================================

def plot_global_feature_importance(mean_shap, rule_freq, lime_freq, feature_names, save_path):
    """Vẽ bar chart so sánh tầm quan trọng feature toàn cục."""
    mean_shap = np.asarray(mean_shap).flatten()
    rule_freq = pd.Series(rule_freq).reindex(feature_names).fillna(0)
    lime_freq = pd.Series(lime_freq).reindex(feature_names).fillna(0)
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


import shap


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
    