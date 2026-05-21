"""
model_io.py
Save and load the trained RandomForest model using Python's built-in pickle.
No external libraries used.
"""

import pickle
import os


def save_model(model, feature_stats, filepath="model.pkl"):
    """
    Serialise the trained model and normalisation stats to disk.

    Parameters
    ----------
    model         : trained RandomForest instance
    feature_stats : list of (min_val, max_val) from min_max_normalize()
    filepath      : destination file path
    """
    payload = {"model": model, "feature_stats": feature_stats}
    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(filepath) // 1024
    print(f"[Model] Saved to '{filepath}'  ({size_kb:,} KB).")


def load_model(filepath="model.pkl"):
    """
    Load the model and feature stats from disk.

    Returns
    -------
    model         : RandomForest instance
    feature_stats : list of (min_val, max_val) tuples
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file '{filepath}' not found. Run train.py first."
        )
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    print(f"[Model] Loaded from '{filepath}'.")
    return payload["model"], payload["feature_stats"]