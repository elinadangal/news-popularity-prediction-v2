"""
data_utils.py
Train/test split and Min-Max normalisation - no external libraries.
"""

import random


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Randomly split (X, y) into training and test sets.

    Parameters
    ----------
    X            : list[list[float]]
    y            : list[str]
    test_size    : float  - fraction kept for testing (default 0.2 = 80/20)
    random_state : int    - seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    random.seed(random_state)
    indices = list(range(len(X)))
    random.shuffle(indices)

    cut = int(len(indices) * (1 - test_size))
    train_idx = indices[:cut]
    test_idx  = indices[cut:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_test  = [y[i] for i in test_idx]

    print(f"[Split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def min_max_normalize(X_train, X_test):
    """
    Scale every feature to [0, 1] using Min-Max normalisation.
    Statistics are computed from X_train only (avoids data leakage).

    Formula:  x_scaled = (x - min) / (max - min)

    Returns
    -------
    X_train_norm  : normalised training set
    X_test_norm   : normalised test set
    feature_stats : list of (min_val, max_val) per feature  <- saved with model
    """
    n_features = len(X_train[0])
    feature_stats = []

    for j in range(n_features):
        col = [row[j] for row in X_train]
        mn  = min(col)
        mx  = max(col)
        feature_stats.append((mn, mx))

    def _scale(row):
        scaled = []
        for j in range(n_features):
            mn, mx = feature_stats[j]
            scaled.append(0.0 if mx == mn else (row[j] - mn) / (mx - mn))
        return scaled

    X_train_norm = [_scale(row) for row in X_train]
    X_test_norm  = [_scale(row) for row in X_test]
    return X_train_norm, X_test_norm, feature_stats


def normalize_single(x, feature_stats):
    """
    Normalise one feature vector using precomputed feature_stats.
    Used at prediction time for user-submitted article data.

    Parameters
    ----------
    x             : list[float]         - raw feature vector
    feature_stats : list[(float,float)] - (min, max) per feature from training

    Returns
    -------
    list[float]  - normalised feature vector
    """
    scaled = []
    for j, val in enumerate(x):
        mn, mx = feature_stats[j]
        scaled.append(0.0 if mx == mn else (val - mn) / (mx - mn))
    return scaled