"""
decision_tree.py
CART Decision Tree classifier built entirely from scratch.
Uses Gini Index for node splitting (as specified in the project proposal).
No external libraries used.
"""

import random


# ── Gini helpers ──────────────────────────────────────────────────────────────

def gini_impurity(labels):
    """
    Gini impurity of a label list.
        Gini(t) = 1 - sum(pi^2)
    """
    total = len(labels)
    if total == 0:
        return 0.0
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


def weighted_gini(left_labels, right_labels):
    """
    Weighted Gini after splitting into left and right.
        Gini_split = (nL/n)*Gini(L) + (nR/n)*Gini(R)
    """
    n = len(left_labels) + len(right_labels)
    if n == 0:
        return 0.0
    return (len(left_labels) / n) * gini_impurity(left_labels) + \
           (len(right_labels) / n) * gini_impurity(right_labels)


def majority_class(labels):
    """Return the most frequent label; break ties alphabetically."""
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


# ── Node ──────────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("feature_index", "threshold", "left", "right", "label")

    def __init__(self):
        self.feature_index = None  # which feature to split on
        self.threshold     = None  # split threshold (go left if <= threshold)
        self.left          = None  # left child node
        self.right         = None  # right child node
        self.label         = None  # set for leaf nodes only


# ── Decision Tree ─────────────────────────────────────────────────────────────

class DecisionTree:
    """
    CART Decision Tree Classifier.

    Parameters
    ----------
    max_depth    : int or None  - maximum depth (None = unlimited)
    min_samples  : int          - minimum samples required to split a node
    max_features : int or None  - features randomly considered per split
                                  (None = all; set to sqrt(m) for Random Forest)
    """

    def __init__(self, max_depth=None, min_samples=2, max_features=None):
        self.max_depth    = max_depth
        self.min_samples  = min_samples
        self.max_features = max_features
        self.root         = None
        self._n_features  = 0

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Build the decision tree from training data."""
        self._n_features = len(X[0]) if X else 0
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        node = _Node()

        # Stopping criteria
        if (len(set(y)) == 1 or
                len(y) < self.min_samples or
                (self.max_depth is not None and depth >= self.max_depth)):
            node.label = majority_class(y)
            return node

        # Feature sub-sampling (for Random Forest)
        all_feats = list(range(self._n_features))
        if self.max_features and self.max_features < self._n_features:
            feat_indices = random.sample(all_feats, self.max_features)
        else:
            feat_indices = all_feats

        best_feat   = None
        best_thresh = None
        best_gini   = float('inf')
        best_left   = []
        best_right  = []

        for fi in feat_indices:
            vals = sorted(set(row[fi] for row in X))
            if len(vals) < 2:
                continue
            # Candidate thresholds = midpoints between consecutive sorted unique values
            thresholds = [(vals[k] + vals[k + 1]) / 2.0 for k in range(len(vals) - 1)]

            for thresh in thresholds:
                left_idx  = [i for i, row in enumerate(X) if row[fi] <= thresh]
                right_idx = [i for i, row in enumerate(X) if row[fi] >  thresh]
                if not left_idx or not right_idx:
                    continue

                g = weighted_gini([y[i] for i in left_idx],
                                   [y[i] for i in right_idx])
                if g < best_gini:
                    best_gini   = g
                    best_feat   = fi
                    best_thresh = thresh
                    best_left   = left_idx
                    best_right  = right_idx

        # No valid split found -> make a leaf
        if best_feat is None:
            node.label = majority_class(y)
            return node

        node.feature_index = best_feat
        node.threshold     = best_thresh
        node.left  = self._build([X[i] for i in best_left],
                                   [y[i] for i in best_left], depth + 1)
        node.right = self._build([X[i] for i in best_right],
                                   [y[i] for i in best_right], depth + 1)
        return node

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_one(self, x):
        """Predict the class for a single sample."""
        node = self.root
        while node.label is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X):
        """Predict class labels for a list of samples."""
        return [self.predict_one(x) for x in X]