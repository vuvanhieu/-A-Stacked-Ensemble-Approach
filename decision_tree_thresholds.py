import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def extract_decision_tree_thresholds(csv_path, feature_cols, target_col, max_depth=3, random_state=42):
    """
    Huấn luyện Decision Tree và trích xuất ngưỡng chia nhánh cho các biến đặc trưng.
    Trả về danh sách các ngưỡng dạng [(feature, threshold)].
    """
    df = pd.read_csv(csv_path)
    X = df[feature_cols]
    y = df[target_col]
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, y)
    thresholds = []
    feature_names = X.columns
    for i in range(clf.tree_.node_count):
        if clf.tree_.children_left[i] != clf.tree_.children_right[i]:
            feature = feature_names[clf.tree_.feature[i]]
            threshold = clf.tree_.threshold[i]
            thresholds.append((feature, threshold))
    return thresholds
