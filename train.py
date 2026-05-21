"""
train.py
Training script for the News Popularity Prediction System.

Steps:
  1. Load  OnlineNewsPopularity.csv  (UCI dataset)
  2. Extract 8 feature columns + shares label
  3. Assign labels  ->  Viral / Average / Flop
  4. 80/20 train-test split
  5. Min-Max normalise features (fit on train only)
  6. Train Random Forest  (50 trees, max_depth=10)
  7. Evaluate on test set
  8. Save model to model.pkl

Usage:
  python train.py

Place OnlineNewsPopularity.csv in the same folder before running.
Download from: https://archive.ics.uci.edu/ml/datasets/online+news+popularity
"""

import os
import time

from csv_reader    import load_dataset, extract_features_and_labels, SELECTED_FEATURE_COLS, LABEL_COL
from data_utils    import train_test_split, min_max_normalize
from random_forest import RandomForest
from evaluation    import print_report, print_confusion_matrix
from model_io      import save_model

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = "OnlineNewsPopularity.csv"
MODEL_PATH   = "model.pkl"

VIRAL_THRESHOLD = 1400   # shares >= 1400         -> Viral
FLOP_THRESHOLD  = 500    # shares <= 500           -> Flop
                          # 500 < shares < 1400    -> Average

N_TREES      = 50
MAX_DEPTH    = 10
MIN_SAMPLES  = 5
RANDOM_STATE = 42


def main():
    start = time.time()
    print("=" * 57)
    print("  News Popularity Prediction System  -  Training")
    print("=" * 57)

    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] '{DATASET_PATH}' not found.")
        print("Download from:")
        print("  https://archive.ics.uci.edu/ml/datasets/online+news+popularity")
        print("Place it in the same folder as this script, then run again.")
        return

    print(f"\n[1/6] Loading '{DATASET_PATH}' ...")
    headers, rows = load_dataset(DATASET_PATH)

    # 2 & 3. Extract features and assign labels
    print(f"\n[2/6] Extracting features and assigning labels ...")
    print(f"      Features used ({len(SELECTED_FEATURE_COLS)}):")
    for col in SELECTED_FEATURE_COLS:
        print(f"        - {col.strip()}")
    print(f"      Viral  threshold : shares >= {VIRAL_THRESHOLD}")
    print(f"      Flop   threshold : shares <= {FLOP_THRESHOLD}")

    X, y = extract_features_and_labels(
        headers, rows,
        feature_cols    = SELECTED_FEATURE_COLS,
        label_col       = LABEL_COL,
        viral_threshold = VIRAL_THRESHOLD,
        flop_threshold  = FLOP_THRESHOLD,
    )

    # 4. Train/test split
    print(f"\n[3/6] Splitting (80% train / 20% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 5. Normalise
    print(f"\n[4/6] Applying Min-Max normalisation ...")
    X_train_norm, X_test_norm, feature_stats = min_max_normalize(X_train, X_test)
    print("      Done.")

    # 6. Train Random Forest
    print(f"\n[5/6] Training Random Forest ({N_TREES} trees, max_depth={MAX_DEPTH}) ...")
    rf = RandomForest(
        n_trees      = N_TREES,
        max_depth    = MAX_DEPTH,
        min_samples  = MIN_SAMPLES,
        random_state = RANDOM_STATE,
    )
    t0 = time.time()
    rf.fit(X_train_norm, y_train)
    print(f"      Training time: {time.time() - t0:.1f}s")

    # 7. Evaluate
    print(f"\n[6/6] Evaluating on test set ({len(X_test):,} samples) ...")
    y_pred = rf.predict(X_test_norm)
    print_report(y_test, y_pred)
    print_confusion_matrix(y_test, y_pred)

    # 8. Save
    save_model(rf, feature_stats, MODEL_PATH)
    print(f"\nTotal time: {time.time() - start:.1f}s")
    print("Run  'python app.py'  to start the web application.\n")


if __name__ == "__main__":
    main()