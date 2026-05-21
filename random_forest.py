"""
random_forest.py
Random Forest Classifier built entirely from scratch.

Algorithm:
  1. For each of N trees:
       a. Draw a bootstrap sample (sample n rows WITH replacement).
       b. Build a DecisionTree; at each node only sqrt(m) random features
          are considered for splitting.
  2. To predict a new sample:
       a. Each tree votes for a class.
       b. Return the class with the most votes (majority voting).

No external libraries used.
"""

import random
from decision_tree import DecisionTree, majority_class


def _isqrt(n):
    """Integer square root without the math module."""
    if n < 0:
        raise ValueError("Square root of negative number.")
    if n == 0:
        return 0
    x = n
    while True:
        x1 = (x + n // x) // 2
        if x1 >= x:
            return x
        x = x1


def _bootstrap_sample(X, y):
    """
    Sample n rows with replacement from (X, y).
    ~36.8% of original samples will not appear in any given bootstrap sample.
    """
    n = len(X)
    indices = [random.randint(0, n - 1) for _ in range(n)]
    return [X[i] for i in indices], [y[i] for i in indices]


def _majority_vote(votes):
    """Return the most frequent label from a list of votes."""
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: (counts[k], k))


class RandomForest:
    """
    Random Forest Classifier.

    Parameters
    ----------
    n_trees      : int        - number of trees to build (default 50)
    max_depth    : int        - max depth of each tree (default 10)
    min_samples  : int        - min samples to split a node (default 5)
    max_features : int|None   - features per split (None -> sqrt of total)
    random_state : int        - seed for reproducibility
    """

    def __init__(self, n_trees=50, max_depth=10,
                 min_samples=5, max_features=None, random_state=42):
        self.n_trees      = n_trees
        self.max_depth    = max_depth
        self.min_samples  = min_samples
        self.max_features = max_features
        self.random_state = random_state
        self.trees        = []

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the Random Forest.

        For each tree:
          1. Create a bootstrap sample from (X, y).
          2. Fit a DecisionTree with random feature sub-sampling at each node.
        """
        random.seed(self.random_state)

        n_features   = len(X[0]) if X else 0
        max_features = self.max_features if self.max_features else _isqrt(n_features)
        max_features = max(1, max_features)

        self.trees = []
        for t in range(1, self.n_trees + 1):
            X_boot, y_boot = _bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth    = self.max_depth,
                min_samples  = self.min_samples,
                max_features = max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            if t % 10 == 0 or t == self.n_trees:
                print(f"  [{t}/{self.n_trees}] trees trained.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict labels for a list of samples using majority voting.

        Returns list[str] of length len(X).
        """
        if not self.trees:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Each tree predicts the full test set -> shape (n_trees, n_samples)
        all_preds = [tree.predict(X) for tree in self.trees]

        # Transpose and vote: for each sample, pick the majority class
        n_samples = len(X)
        return [
            _majority_vote([all_preds[t][i] for t in range(len(self.trees))])
            for i in range(n_samples)
        ]

    def predict_one(self, x):
        """Predict the class for a single feature vector."""
        if not self.trees:
            raise RuntimeError("Model not trained. Call fit() first.")
        votes = [tree.predict_one(x) for tree in self.trees]
        return _majority_vote(votes)