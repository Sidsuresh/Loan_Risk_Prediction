# 🏦 Loan Default Risk Prediction App

A full-stack Machine Learning application that predicts the probability of a loan defaulting based on borrower attributes. This project demonstrates a complete ML pipeline, from data preprocessing and outlier handling to a live, interactive Streamlit dashboard.

## 🚀 Live Demo

[Insert Link to your Streamlit Cloud Deployment Here]

## 🛠️ Features

- **Real-time Risk Assessment:** Input borrower data to get an instant probability of default.
- **Dynamic UI:** Interactive forms with human-friendly labels (e.g., "Fractional" instead of "F").
- **Outlier Robustness:** Automated clipping of input values based on the Interquartile Range (IQR) used during training.
- **Visual Risk Gauges:** Color-coded results (Green/Orange/Red) based on the calculated risk percentage.

## 📊 The Machine Learning Pipeline

1.  **Exploratory Data Analysis (EDA):** Identified key features like FICO scores, Debt-to-Income (DTI) ratios, and loan terms.
2.  **Preprocessing:** \* Label Encoding for categorical variables.
    - Standard Scaling for numerical features.
    - Custom Feature Engineering (e.g., Average FICO score, Issue Year extraction).
3.  **Model:** Gradient Boosting Classifier optimized for high-precision risk detection.
4.  **Deployment:** Model and Scaler bundled via `cloudpickle` and served through Streamlit.

## 💻 Tech Stack

- **Language:** Python 3.11+
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Deployment:** Streamlit, Poetry (Dependency Management)
- **Serialization:** Cloudpickle

## 📂 Project Structure

```text
├── app/
│   └── main.py          # Streamlit Application logic
├── models/
│   └── deployment_bundle.pkl  # Pickled Model, Scaler, and Encoders
├── app_config.json      # Metadata for UI constraints and categories
├── pyproject.toml       # Poetry dependency configuration
└── README.md
```
