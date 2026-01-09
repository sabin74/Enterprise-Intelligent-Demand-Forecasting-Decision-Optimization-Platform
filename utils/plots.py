# plots.py

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------
# DAILY TOTAL SALES TREND
# ---------------------------------
def plot_daily_sales_trend(df, date_col="date", sales_col="sales"):
    """
    Plot total daily predicted sales
    """
    daily = (
        df.groupby(date_col)[sales_col]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily[date_col], daily[sales_col])
    ax.set_title("Daily Total Predicted Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)

    return fig


# ---------------------------------
# STORE-LEVEL SALES TREND
# ---------------------------------
def plot_store_sales_trend(df, store_nbr, date_col="date", sales_col="sales"):
    """
    Plot sales trend for a specific store
    """
    store_df = df[df["store_nbr"] == store_nbr]

    daily = (
        store_df.groupby(date_col)[sales_col]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily[date_col], daily[sales_col])
    ax.set_title(f"Store {store_nbr} – Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)

    return fig


# ---------------------------------
# FAMILY-LEVEL SALES TREND
# ---------------------------------
def plot_family_sales_trend(df, family, date_col="date", sales_col="sales"):
    """
    Plot sales trend for a specific product family
    """
    fam_df = df[df["family"] == family]

    daily = (
        fam_df.groupby(date_col)[sales_col]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily[date_col], daily[sales_col])
    ax.set_title(f"Family '{family}' – Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True)

    return fig


# ---------------------------------
# SALES DISTRIBUTION
# ---------------------------------
def plot_sales_distribution(df, sales_col="sales"):
    """
    Histogram of predicted sales
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df[sales_col], bins=50)
    ax.set_title("Distribution of Predicted Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    return fig


# ---------------------------------
# TOP STORES BY SALES
# ---------------------------------
def plot_top_stores(df, top_n=10, sales_col="sales"):
    """
    Bar chart of top-N stores by total sales
    """
    top_stores = (
        df.groupby("store_nbr")[sales_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    top_stores.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Stores by Total Sales")
    ax.set_xlabel("Store Number")
    ax.set_ylabel("Sales")
    ax.grid(True)

    return fig

