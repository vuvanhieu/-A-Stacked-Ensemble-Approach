# configs.py

import os
from pathlib import Path
# ==== Thư mục dữ liệu, output, backup ====
DATA_FILE = "Bank_Customer_Churn.csv"
# Có thể bổ sung dataset khác để kiểm tra tính tổng quát
# DATA_FILE_2 = "Telco_Customer_Churn.csv"
BACKUP_DIRNAME = "backup"
MULTITASK_DNN_DIRNAME = "Multitask_DNN"
BASELINE_DIRNAME = "Baselines"
EXPLANATION_COMPARISON_DIRNAME = "Explanation_Comparison"
CHURN_ALERT_DIRNAME = "churn_risk_alert"

# ==== Tên file xuất kết quả chuẩn hóa ====
MULTITASK_GRIDSEARCH_CSV = "gridsearch_multitask_results.csv"
BASELINE_COMPARISON_CSV = "baseline_comparison.csv"
COMPARE_MODELS_SUMMARY_CSV = "compare_models_summary.csv"
THRESHOLDS_CSV = "decision_tree_thresholds.csv"
EXPLANATION_RESULTS_CSV = "explanation_results.csv"
HIGH_RISK_CUSTOMERS_CSV = "high_risk_customers_with_reasons.csv"
HIGH_RISK_REASON_PNG = "high_risk_reason_barplot.png"

# ==== Cross-validation config ====
N_FOLDS = 5         # Số fold cho cross-validation
N_REPEATS = 3       # Số lần lặp lại cross-validation (nếu dùng RepeatedKFold)

# ==== Random seed dùng chung ====
RANDOM_STATE = 42

# ==== Risk Threshold Config (dùng cho rule‑based explanations) ====
LOW_CREDIT_THRESHOLD = 580
HIGH_CREDIT_THRESHOLD = 700
LOW_BALANCE_THRESHOLD = 1000
HIGH_BALANCE_THRESHOLD = 100000
NEW_CUSTOMER_TENURE = 2
SENIOR_AGE_THRESHOLD = 60
INACTIVE_MEMBER_FLAG = 0
LOW_PRODUCT_COUNT = 1
HIGH_PRODUCT_COUNT = 3
NO_CREDIT_CARD_FLAG = 0

# ==== Font size config for plots ====
BASE_FONT_SIZE = 14
TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
SHAP_NSAMPLES = 30
SHAP_BACKGROUND_SIZE = 200  # Số lượng background samples cho SHAP
# SHAP_NSAMPLES = 10
# SHAP_BACKGROUND_SIZE = 50  # Số lượng background samples cho SHAP

EXPLANATION_TOP_K = 5
EXPLANATION_MAX_SAMPLES = 5  # Số lượng mẫu tối đa cho so sánh giải thích (SHAP/LIME/rule)
JACCARD_BOXPLOT_COLORS = ['#ff9999', '#66b3ff', '#99ff99']

# ==== Pipeline configuration ====
# Chọn pipeline: "tree" hoặc "neural" – dùng để quyết định preprocessing
PIPELINE_TYPE = "tree"      # mặc định là tree (không scale, chỉ label encode)

# ==== Cost / Profit parameters ====
COST_FP = 10      # chi phí cho false positive (gửi ưu đãi sai)
COST_FN = 50      # chi phí cho false negative (mất khách hàng)
PROFIT_TP = 200   # lợi nhuận khi giữ chân thành công
PROFIT_TN = 0     # lợi nhuận khi không churn (không tốn chi phí)

# ==== Đường dẫn project ====
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = Path(PROJECT_PATH) / "OUTPUT"


# ==== Hàm tiện ích ====
def get_data_path(filename=DATA_FILE):
    return os.path.join(PROJECT_PATH, filename)

def get_output_subdir(*subdirs):
    path = os.path.join(OUTPUT_DIR, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path