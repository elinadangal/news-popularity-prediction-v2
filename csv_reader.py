"""
csv_reader.py
Reads the UCI Online News Popularity CSV file without any external libraries.

"""

import csv  


#  The 8 selected feature columns  
SELECTED_FEATURE_COLS = [
    " n_tokens_title",
    " n_tokens_content",
    " average_token_length",
    " num_hrefs",
    " num_imgs",
    " num_videos",
    " global_subjectivity",
    " global_sentiment_polarity",
]

# Human-readable names (same order, used for display)
FEATURE_DISPLAY_NAMES = [
    "n_tokens_title",
    "n_tokens_content",
    "average_token_length",
    "num_hrefs",
    "num_imgs",
    "num_videos",
    "global_subjectivity",
    "global_sentiment_polarity",
]

LABEL_COL = " shares"   # target column 


def load_dataset(filepath):
    """
    Load OnlineNewsPopularity.csv.

    Returns
    headers : list[str]  - raw column names exactly as in the CSV
    rows    : list[list] - every data row as a list of strings
    """
    headers = []
    rows = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                headers = row          # keep original names 
            else:
                rows.append(row)
    print(f"[CSV] Loaded {len(rows):,} rows, {len(headers)} columns.")
    return headers, rows


def get_col_index(headers, col_name):
    """Return the index of a column by its exact name. Raises ValueError if missing."""
    for i, h in enumerate(headers):
        if h == col_name:
            return i
    raise ValueError(f"Column '{col_name}' not found in headers.")


def assign_label(shares, viral_threshold=1400, flop_threshold=500):
    """
    Convert a share count into one of three class labels.
        shares >= viral_threshold  -> "Viral"
        shares <= flop_threshold   -> "Flop"
        otherwise                  -> "Average"
    """
    if shares >= viral_threshold:
        return "Viral"
    elif shares <= flop_threshold:
        return "Flop"
    else:
        return "Average"


def extract_features_and_labels(headers, rows,
                                  feature_cols=None,
                                  label_col=LABEL_COL,
                                  viral_threshold=1400,
                                  flop_threshold=500):
    """
    Extract numerical feature vectors and class labels from raw CSV rows.

    Parameters
    ----------
    headers         : list[str]        - from load_dataset()
    rows            : list[list[str]]  - from load_dataset()
    feature_cols    : list[str]        - columns to use (default: SELECTED_FEATURE_COLS)
    label_col       : str              - column containing the share count
    viral_threshold : int
    flop_threshold  : int

    Returns
    -------
    X : list[list[float]]  - feature matrix  (n_samples x n_features)
    y : list[str]          - labels ("Viral", "Average", or "Flop")
    """
    if feature_cols is None:
        feature_cols = SELECTED_FEATURE_COLS

    feat_indices = [get_col_index(headers, c) for c in feature_cols]
    label_index  = get_col_index(headers, label_col)

    X, y = [], []
    skipped = 0
    for row in rows:
        try:
            features = [float(row[idx]) for idx in feat_indices]
            shares   = float(row[label_index])
        except (ValueError, IndexError):
            skipped += 1
            continue
        X.append(features)
        y.append(assign_label(shares, viral_threshold, flop_threshold))

    if skipped:
        print(f"[CSV] Skipped {skipped} malformed rows.")

    counts = {}
    for lbl in y:
        counts[lbl] = counts.get(lbl, 0) + 1
    print("[CSV] Label distribution:")
    for lbl in sorted(counts):
        print(f"        {lbl}: {counts[lbl]:,}  ({100 * counts[lbl] / len(y):.1f}%)")

    return X, y