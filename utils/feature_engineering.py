# feature_engineering.py

import gc
import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
LAGS  = [1, 7, 14, 28]
ROLLS = [7, 14, 28]
MAX_LAG = max(LAGS)


# -----------------------------
# MEMORY OPTIMIZATION
# -----------------------------
def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


# -----------------------------
# FEATURE GENERATORS
# -----------------------------
def generate_sales_features(history, current, lags, rolls):
    for lag in lags:
        current[f"sales_lag_{lag}"] = (
            history.groupby(["store_nbr", "family"])["sales"]
            .shift(lag)
            .reindex(current.index)
        )

    for r in rolls:
        roll = (
            history.groupby(["store_nbr", "family"])["sales"]
            .rolling(r)
            .agg(["mean", "std"])
            .reset_index(level=[0, 1], drop=True)
        )
        current[f"sales_roll_mean_{r}"] = roll["mean"].reindex(current.index)
        current[f"sales_roll_std_{r}"]  = roll["std"].reindex(current.index)

    return current


def generate_promo_features(history, current, lags, rolls):
    for lag in lags:
        current[f"promo_lag_{lag}"] = (
            history.groupby(["store_nbr", "family"])["onpromotion"]
            .shift(lag)
            .reindex(current.index)
        )

    for r in rolls:
        rolling = history.groupby(["store_nbr", "family"])["onpromotion"].rolling(r)
        current[f"promo_roll_sum_{r}"] = rolling.sum().reset_index(level=[0, 1], drop=True)
        current[f"promo_freq_{r}"]     = rolling.mean().reset_index(level=[0, 1], drop=True)

    current["promo_flag"] = (current["onpromotion"] > 0).astype(int)
    return current


def generate_oil_features(history, current, lags):
    for lag in lags:
        current[f"oil_lag_{lag}"] = (
            history.groupby("store_nbr")["dcoilwtico"]
            .shift(lag)
            .reindex(current.index)
        )
    return current


# -----------------------------
# MAIN DEPLOYMENT FUNCTION
# -----------------------------
def build_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predict_fn,
    train_features: list
):
    """
    Parameters
    ----------
    train_df : training dataframe with true sales
    test_df  : test_features.parquet
    predict_fn : function(X) -> predicted sales (original scale)
    train_features : exact training feature list

    Returns
    -------
    test_final : dataframe with predictions
    """

    train_df = reduce_mem_usage(train_df)
    test_df  = reduce_mem_usage(test_df)

    # Build initial history
    history_df = train_df[
        train_df["date"] >= (train_df["date"].max() - pd.Timedelta(days=MAX_LAG + 30))
    ][["store_nbr", "family", "date", "sales", "onpromotion", "dcoilwtico"]].copy()

    history_df = history_df.sort_values(
        ["store_nbr", "family", "date"]
    ).reset_index(drop=True)

    test_preds = []
    test_dates = sorted(test_df["date"].unique())

    for current_date in test_dates:

        test_day = test_df[test_df["date"] == current_date].copy()
        temp_history = pd.concat([history_df, test_day], ignore_index=True)

        # Feature engineering
        test_day = generate_sales_features(temp_history, test_day, LAGS, ROLLS)
        test_day = generate_promo_features(temp_history, test_day, [1, 7], ROLLS)
        test_day = generate_oil_features(temp_history, test_day, [7, 14, 28])

        # Safe filling
        sales_cols = [c for c in test_day.columns if "sales_" in c]
        test_day[sales_cols] = test_day[sales_cols].fillna(0)
        test_day = test_day.ffill().bfill()

        # Predict sales
        test_day["sales"] = predict_fn(test_day[train_features])

        # Update history
        history_df = pd.concat([history_df, test_day], ignore_index=True)
        test_preds.append(test_day)

    gc.collect()
    return pd.concat(test_preds).reset_index(drop=True)

