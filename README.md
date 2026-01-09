# 📦 Enterprise Intelligent Demand Forecasting & Decision Optimization Platform

An end‑to‑end **enterprise‑grade demand forecasting system** designed to predict daily sales at scale using **Machine Learning, Deep Learning, and Time Series modeling**, followed by **decision‑oriented optimization** for business use cases such as inventory planning and operations.

This project is inspired by real‑world retail forecasting challenges and is structured to reflect **production‑ready ML pipelines** used in industry.

---

## 🚀 Project Objectives

- Forecast **daily product‑level demand** across multiple stores
- Incorporate **promotions, holidays, oil prices, and seasonality**
- Compare **baseline, ML, and DL models** systematically
- Build a **robust inference pipeline** for unseen test data
- Enable **decision optimization** using forecast outputs
- Follow **enterprise ML best practices** (modularity, reproducibility, scalability)

---

## 🧠 Key Features

✅ End‑to‑end forecasting pipeline (train → validate → test → inference)  
✅ Rich **time‑series feature engineering** (lags, rolling stats, calendars)  
✅ Multiple model families:
- Statistical baselines
- Classical ML (Tree‑based)
- Deep Learning (LSTM / GRU)

✅ Log‑scale training with **RMSLE optimization**  
✅ Prediction interval estimation  
✅ Model artifact versioning (scalers, encoders, models)  
✅ Kaggle‑ready submission generation  

---

## 🏗️ Project Architecture

```
Enterprise-Intelligent-Demand-Forecasting-Decision-Optimization-Platform/
│
├── app/                    # Core pipeline scripts
│   ├── train.py            # Model training entry point
│   ├── inference.py        # Test / production inference
│   └── evaluate.py         # Model evaluation
│
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned & feature‑engineered data
│   └── external/           # Holidays, oil prices, metadata
│
├── notebooks/              # Jupyter notebooks (step‑by‑step development)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_deep_learning.ipynb
│   ├── 06_ensembling.ipynb
│   └── 07_inference_pipeline.ipynb
│
├── models/
│   ├── baselines/
│   ├── ml_models/
│   └── deep_learning/
│
├── outputs/
│   ├── predictions/        # CSV / Parquet outputs
│   ├── submission/         # Kaggle submission files
│   └── evaluation/         # Metrics & plots
│
├── utils/
│   ├── data_utils.py       # Data loading & cleaning
│   ├── feature_utils.py    # Feature engineering helpers
│   ├── model_utils.py      # Training & saving models
│   └── metrics.py          # Evaluation metrics
│
├── config/
│   └── config.yaml         # Centralized configuration
│
├── requirements.txt
└── README.md
```

---

## 📊 Data Description

The dataset consists of **daily sales records** with the following dimensions:

- **Store ID**
- **Product family**
- **Date**
- **Sales volume**
- **Promotions (onpromotion)**
- **Holidays & events**
- **Oil prices**

📌 Target Variable:
- `sales` (modeled using `log1p(sales)` for stability)

---

## 🧪 Feature Engineering

- Lag features: `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_28`
- Rolling statistics:
  - 7 / 14 / 28 day mean
  - Rolling std
- Calendar features:
  - Day of week
  - Week of year
  - Month
  - Is weekend
- Event‑based features:
  - Holiday indicators
  - Promotion flags

---

## 🤖 Models Implemented

### 1️⃣ Baseline Models

- Naive (Last value)
- Moving Average (7 / 14 / 28 days)

### 2️⃣ Machine Learning Models

- LightGBM
- XGBoost
- CatBoost

### 3️⃣ Deep Learning Models

- LSTM (Sliding window approach)
- GRU

📌 All DL models are trained on **log‑scaled targets** and evaluated using RMSLE.

---

## 📐 Evaluation Metrics

- **RMSLE** (Primary)
- MAE
- Store‑level error analysis
- Product‑level error analysis

---

## 📤 Inference & Submission

The inference pipeline:

- Loads trained models & preprocessing artifacts
- Generates predictions for unseen test data
- Applies inverse log transformation
- Clips negative values
- Saves outputs as:
  - CSV
  - Parquet
- Produces **Kaggle‑ready submission files**

---

## 📦 Installation

```bash
git clone https://github.com/sabin74/Enterprise-Intelligent-Demand-Forecasting-Decision-Optimization-Platform.git
cd Enterprise-Intelligent-Demand-Forecasting-Decision-Optimization-Platform
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train Models
```bash
python app/train.py
```

### Run Inference
```bash
python app/inference.py
```

---

## 🧩 Decision Optimization (Planned / Extension)

- Safety stock calculation
- Reorder point estimation
- Promotion sensitivity analysis
- Demand‑driven inventory recommendations

---

## 🧠 Skills Demonstrated

- Time Series Forecasting
- Feature Engineering at Scale
- ML & Deep Learning Model Design
- Pipeline Engineering
- Model Evaluation & Validation
- Production‑style Inference Design

---

## 🔮 Future Improvements

- Probabilistic forecasting (Quantile loss)
- Model stacking & weighted ensembling
- SHAP‑based explainability
- API deployment (FastAPI)
- Real‑time forecasting support

---

## 👤 Author

**Sabin Lamsal**  
Machine Learning & Data Science Enthusiast  
Focused on building scalable, real‑world AI systems

---

⭐ If you find this project useful, consider giving it a star!

