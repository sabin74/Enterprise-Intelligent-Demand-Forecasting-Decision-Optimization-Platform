# predictor.py

import numpy as np
import joblib
import xgboost as xgb


# ---------------------------------
# LOAD ENSEMBLE BUNDLE
# ---------------------------------
def load_bundle(bundle_path: str):
    """
    Load the final ensemble model bundle
    """
    return joblib.load(bundle_path)


# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict_from_bundle(X_raw, bundle):
    """
    Parameters
    ----------
    X_raw : pd.DataFrame
        Feature dataframe with EXACT training features
    bundle : dict
        Loaded ensemble bundle

    Returns
    -------
    np.ndarray
        Predicted sales (original scale)
    """

    # Target encoding (RF + XGB)
    X_te = bundle["target_encoder"].transform(X_raw)

    # Base model predictions (log-scale)
    preds = np.column_stack([
        bundle["rf_model"].predict(X_te),
        bundle["xgb_model"].predict(xgb.DMatrix(X_te)),
        bundle["lgb_model"].predict(
            X_raw,
            num_iteration=bundle["lgb_model"].best_iteration
        ),
        bundle["cat_model"].predict(X_raw),
    ])

    # Meta-model (stacking)
    y_log = bundle["meta_model"].predict(preds)

    # Bias correction
    y_log += bundle["bias"]

    # Zero-sales handling
    y_log = np.where(y_log < bundle["zero_threshold"], 0, y_log)

    # Back to original scale
    return np.expm1(y_log)

