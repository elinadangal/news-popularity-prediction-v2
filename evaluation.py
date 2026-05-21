"""
evaluation.py
Classification metrics computed entirely from scratch.
Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix.
No external libraries used.
"""


def accuracy(y_true, y_pred):
    """Overall accuracy = correctly predicted / total."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def build_confusion_matrix(y_true, y_pred, classes):
    """
    Build a confusion matrix as a nested dict.
        matrix[actual][predicted] = count
    """
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for actual, predicted in zip(y_true, y_pred):
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    return matrix


def precision_recall_f1_per_class(y_true, y_pred, classes):
    """
    Compute Precision, Recall, F1 for each class.

    For class C:
        TP      = correctly predicted as C
        FP      = other classes wrongly predicted as C
        FN      = C samples wrongly predicted as another class
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * Precision * Recall / (Precision + Recall)

    Returns dict[class -> {precision, recall, f1, support}]
    """
    cm = build_confusion_matrix(y_true, y_pred, classes)
    results = {}
    for cls in classes:
        tp      = cm[cls][cls]
        fp      = sum(cm[other][cls] for other in classes if other != cls)
        fn      = sum(cm[cls][other] for other in classes if other != cls)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[cls] = {"precision": precision, "recall": recall,
                        "f1": f1, "support": support}
    return results


def macro_f1(per_class_metrics):
    """Unweighted (macro) average F1 across all classes."""
    scores = [v["f1"] for v in per_class_metrics.values()]
    return sum(scores) / len(scores) if scores else 0.0


def print_report(y_true, y_pred):
    """Print a full classification report and return a summary dict."""
    classes   = sorted(set(y_true) | set(y_pred))
    acc       = accuracy(y_true, y_pred)
    per_class = precision_recall_f1_per_class(y_true, y_pred, classes)
    m_f1      = macro_f1(per_class)
    total     = len(y_true)

    W = 58
    print(f"\n{'=' * W}")
    print(f"{'Classification Report':^{W}}")
    print(f"{'=' * W}")
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>9} {'F1-Score':>9} {'Support':>9}")
    print(f"{'-' * W}")
    for cls in classes:
        m = per_class[cls]
        print(f"{cls:<14} {m['precision']:>10.4f} {m['recall']:>9.4f} "
              f"{m['f1']:>9.4f} {m['support']:>9}")
    print(f"{'-' * W}")
    print(f"{'Accuracy':<14} {'':>10} {'':>9} {acc:>9.4f} {total:>9}")
    print(f"{'Macro Avg F1':<14} {'':>10} {'':>9} {m_f1:>9.4f}")
    print(f"{'=' * W}\n")

    return {"accuracy": acc, "per_class": per_class, "macro_f1": m_f1}


def print_confusion_matrix(y_true, y_pred):
    """Pretty-print the confusion matrix."""
    classes = sorted(set(y_true) | set(y_pred))
    cm      = build_confusion_matrix(y_true, y_pred, classes)
    col_w   = 10

    print("Confusion Matrix  (rows = Actual, columns = Predicted)")
    print(" " * 14 + "".join(f"{c:>{col_w}}" for c in classes))
    print("-" * (14 + col_w * len(classes)))
    for actual in classes:
        row_str = f"{actual:<14}" + "".join(
            f"{cm[actual][pred]:>{col_w}}" for pred in classes
        )
        print(row_str)
    print()