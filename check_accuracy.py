from model_io import load_model
from csv_reader import load_dataset, extract_features_and_labels, SELECTED_FEATURE_COLS, LABEL_COL
from data_utils import train_test_split, min_max_normalize
from evaluation import print_report, print_confusion_matrix

# Load model
model, feature_stats = load_model("model.pkl")

# Load dataset
headers, rows = load_dataset("OnlineNewsPopularity.csv")
X, y = extract_features_and_labels(headers, rows,
                                    feature_cols=SELECTED_FEATURE_COLS,
                                    label_col=LABEL_COL)

# Split and normalize (same seed = same test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, X_test_norm, _ = min_max_normalize(X_train, X_test)

# Evaluate
y_pred = model.predict(X_test_norm)
print_report(y_test, y_pred)
print_confusion_matrix(y_test, y_pred)