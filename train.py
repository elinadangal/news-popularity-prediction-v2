"""
train.py
Fast training script for the News Popularity Prediction System.

Changes for speed:
  - Uses 8,000 samples per class (balanced) instead of full 39,644 rows
  - Uses 20 trees instead of 50
  - Uses max_depth=8 instead of 10
  - Balanced thresholds so Flop class is well represented:
      Viral   : shares >= 1400
      Average : 700 < shares < 1400
      Flop    : shares <= 700

Expected training time: 2 to 5 minutes on a standard laptop.
"""

import os
import time
import random

from csv_reader    import load_dataset, extract_features_and_labels, SELECTED_FEATURE_COLS, LABEL_COL
from data_utils    import train_test_split, min_max_normalize
from random_forest import RandomForest
from evaluation    import print_report, print_confusion_matrix
from model_io      import save_model

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = "OnlineNewsPopularity.csv"
MODEL_PATH   = "model.pkl"

# Balanced thresholds so Flop class has enough samples
VIRAL_THRESHOLD = 1600  # shares >= 1600       -> Viral
FLOP_THRESHOLD  = 800    # shares <= 800        -> Flop
                          # 800 < shares < 1600 -> Average

# Faster hyperparameters
N_TREES      = 20        # reduced from 50
MAX_DEPTH    = 8         # reduced from 10
MIN_SAMPLES  = 5
RANDOM_STATE = 42

# Balanced sample size per class
SAMPLE_PER_CLASS = 8000


def balanced_sample(X, y, per_class, seed=42):
    """
    Take up to `per_class` samples from each class so all three classes
    are equally represented and training is fast.
    """
    random.seed(seed)
    classes = list(set(y))
    X_out, y_out = [], []

    for cls in classes:
        indices = [i for i, label in enumerate(y) if label == cls]
        if len(indices) > per_class:
            indices = random.sample(indices, per_class)
        X_out.extend([X[i] for i in indices])
        y_out.extend([y[i] for i in indices])

    # Shuffle the combined result
    combined = list(zip(X_out, y_out))
    random.shuffle(combined)
    X_out, y_out = zip(*combined)
    return list(X_out), list(y_out)


def main():
    start = time.time()
    print("=" * 57)
    print("  News Popularity Prediction   ")
    print("=" * 57)

    # 1. Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] '{DATASET_PATH}' not found.")
        print("Download from:")
        print("  https://archive.ics.uci.edu/ml/datasets/online+news+popularity")
        return

    print(f"\n[1/6] Loading '{DATASET_PATH}' ...")
    headers, rows = load_dataset(DATASET_PATH)

    # 2. Extract features and assign labels with balanced thresholds
    print(f"\n[2/6] Extracting features and assigning labels ...")
    print(f"      Viral  : shares >= {VIRAL_THRESHOLD}")
    print(f"      Flop   : shares <= {FLOP_THRESHOLD}")
    print(f"      Average: {FLOP_THRESHOLD} < shares < {VIRAL_THRESHOLD}")

    X, y = extract_features_and_labels(
        headers, rows,
        feature_cols    = SELECTED_FEATURE_COLS,
        label_col       = LABEL_COL,
        viral_threshold = VIRAL_THRESHOLD,
        flop_threshold  = FLOP_THRESHOLD,
    )

    # 3. Balanced sampling
    print(f"\n[3/6] Balanced sampling ({SAMPLE_PER_CLASS} per class) ...")
    X, y = balanced_sample(X, y, per_class=SAMPLE_PER_CLASS, seed=RANDOM_STATE)
    counts = {}
    for lbl in y:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"      Total samples: {len(X):,}")
    for lbl in sorted(counts):
        print(f"        {lbl}: {counts[lbl]:,}  ({100 * counts[lbl] / len(y):.1f}%)")

    # 4. Train/test split
    print(f"\n[4/6] Splitting (80% train / 20% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 5. Normalise
    print(f"\n[5/6] Applying Min-Max normalisation ...")
    X_train_norm, X_test_norm, feature_stats = min_max_normalize(X_train, X_test)
    print("      Done.")

    # 6. Train Random Forest
    print(f"\n[6/6] Training Random Forest ({N_TREES} trees, max_depth={MAX_DEPTH}) ...")
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
    print(f"\nEvaluating on test set ({len(X_test):,} samples) ...")
    y_pred = rf.predict(X_test_norm)
    print_report(y_test, y_pred)
    print_confusion_matrix(y_test, y_pred)

    # 8. Save model
    save_model(rf, feature_stats, MODEL_PATH)
    print(f"\nTotal time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()