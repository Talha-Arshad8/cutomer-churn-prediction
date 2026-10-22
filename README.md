# cutomer-churn-prediction
Customer churn prediction using machine learning and Streamlit

# Customer Churn Prediction

A machine learning project that analyzes customer churn and predicts whether a customer is likely to leave a telecom service. The project includes an interactive Streamlit dashboard for customer exploration, churn analysis, model comparison, prediction, and retention insights.

## Project Features

- Customer Explorer with search and filters
- Churn Dashboard and exploratory analysis
- Comparison of Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest
- Model evaluation using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Threshold Analysis
- Individual Customer Churn Prediction
- Retention Dashboard

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The original dataset contains 7,043 customer records and 21 columns. The target variable is `Churn`.

## Machine Learning Models

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn

## Run Locally

Install the required packages:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

Run the dashboard:

```bash
streamlit run app.py
```

The trained model file (`best_churn_model.pkl`) and feature list (`model_features.pkl`) must be available in the project directory.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── best_churn_model.pkl
├── model_features.pkl
├── model_comparison.csv
└── README.md
```

## Disclaimer

This project is for educational and portfolio purposes. Predictions are estimates and should not be treated as guaranteed customer behavior.

