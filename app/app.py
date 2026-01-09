# app.py

import streamlit as st
import pandas as pd
import sys
import os

# Add utils folder to path
sys.path.append(os.path.abspath("../utils"))

from feature_engineering import build_test_features
from predictor import load_bundle, predict_from_bundle
from plots import (
    plot_daily_sales_trend,
    plot_store_sales_trend,
    plot_family_sales_trend,
    plot_sales_distribution
)

st.set_page_config(page_title="Enterprise Demand Forecasting", layout="wide")
st.title("📈 Enterprise Demand Forecasting App")

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def get_bundle():
    bundle_path = "../models/ensemble-stacking/final_ensemble_model.pkl"
    return load_bundle(bundle_path)

bundle = get_bundle()

# -----------------------------
# Load training data (for history)
# -----------------------------
@st.cache_data
def load_train():
    return pd.read_parquet("../data/features/train_features.parquet")

train_df = load_train()

# Training features
DROP_COLS = ["id", "date", "sales", "sales_log"]
TRAIN_FEATURES = [c for c in train_df.columns if c not in DROP_COLS]

# -----------------------------
# File uploader
# -----------------------------
st.sidebar.header("Upload Test Feature File")
uploaded_file = st.sidebar.file_uploader(
    "Upload your test_feature.parquet",
    type=["parquet"]
)

if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully!")

    test_df = pd.read_parquet(uploaded_file)
    st.write("Test feature preview:")
    st.dataframe(test_df.head())

    # -----------------------------
    # Run Prediction
    # -----------------------------
    if st.button("🚀 Run Prediction"):
        st.info("Running predictions. Please wait...")

        test_final = build_test_features(
            train_df=train_df,
            test_df=test_df,
            predict_fn=lambda X: predict_from_bundle(X, bundle),
            train_features=TRAIN_FEATURES
        )

        st.success("✅ Prediction completed!")
        st.dataframe(test_final.head())

        # -----------------------------
        # Download submission
        # -----------------------------
        submission = test_final[["id", "sales"]].rename(columns={"sales": "sales_pred"})
        st.download_button(
            label="⬇️ Download Submission CSV",
            data=submission.to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv"
        )

        # -----------------------------
        # Visualizations
        # -----------------------------
        st.subheader("Visualizations")

        # Daily trend
        st.markdown("### Daily Total Sales Trend")
        fig1 = plot_daily_sales_trend(test_final, sales_col="sales")
        st.pyplot(fig1)

        # Store-level trend
        st.markdown("### Store-Level Sales Trend")
        store_list = test_final["store_nbr"].unique()
        selected_store = st.selectbox("Select Store", store_list)
        fig2 = plot_store_sales_trend(test_final, selected_store, sales_col="sales")
        st.pyplot(fig2)

        # Family-level trend
        st.markdown("### Product Family Sales Trend")
        family_list = test_final["family"].unique()
        selected_family = st.selectbox("Select Product Family", family_list)
        fig3 = plot_family_sales_trend(test_final, selected_family, sales_col="sales")
        st.pyplot(fig3)

        # Sales distribution
        st.markdown("### Sales Distribution")
        fig4 = plot_sales_distribution(test_final, sales_col="sales")
        st.pyplot(fig4)

else:
    st.warning("Please upload a test_feature.parquet file to start prediction.")

