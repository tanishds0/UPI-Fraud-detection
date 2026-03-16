# UPI Fraud Detection System

## Project Overview
This project is an end-to-end Machine Learning system to detect **fraudulent UPI transactions**. It includes:
- Synthetic dataset generation (`data/transactions.csv`)
- EDA notebook (`notebooks/EDA.ipynb`)
- Modular ML pipeline (preprocessing, feature engineering, training, evaluation)
- A Streamlit web app to check whether a transaction is **Fraudulent** or **Genuine**

## Problem Statement
UPI transactions are high-volume and real-time. Detecting suspicious behavior helps reduce financial losses and improves trust. This project trains ML models to predict the probability that a transaction is fraudulent.

## Tech Stack
- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Streamlit**
- **Joblib**

## Project Architecture
```
upi-fraud-detection/
├── data/
│   └── transactions.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   └── fraud_model.pkl            # generated after training
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1) Install Python
Make sure **Python 3.10+** is installed and available on PATH:
```bash
python --version
```

### 2) Create a virtual environment and install dependencies
From the project root (`upi-fraud-detection/`):

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```

Install deps:
```bash
pip install -r requirements.txt
```

### 3) Generate dataset (10,000 rows)
```bash
python -m src.generate_data
```
This creates/overwrites `data/transactions.csv`.

### 4) Train models and save the best one
```bash
python -m src.train_model
```
This trains:
- Logistic Regression
- Random Forest
- Gradient Boosting

It selects the best model (by **F1**, then **ROC-AUC**) and saves:
- `models/fraud_model.pkl`

### 5) Evaluate the saved model
```bash
python -m src.evaluate_model
```
Outputs:
- Printed metrics + classification report
- `models/confusion_matrix.png`
- `models/roc_curve.png`

### 6) Run the Streamlit web app
```bash
streamlit run app/streamlit_app.py
```

## Streamlit App Inputs
The app lets you enter:
- transaction amount
- transaction type
- device type
- number of previous transactions
- account age (days)

The app loads `models/fraud_model.pkl`, predicts fraud probability, and displays:
- **Fraudulent Transaction** (probability ≥ 0.5)
- **Genuine Transaction** (probability < 0.5)

## Example Screenshots
Add screenshots after running the project:
- `docs/streamlit_home.png`
- `docs/prediction_result.png`

(Create a `docs/` folder if you’d like to store images.)

## Future Improvements
- Use a real-world dataset and handle class imbalance more robustly (SMOTE/threshold tuning)
- Add transaction time-series behavior features per user
- Model monitoring (data drift, concept drift)
- Add explainability (SHAP) to show top contributing factors
- Deploy to cloud (Docker, CI/CD, model registry)

