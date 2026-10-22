# Customer Churn Prediction

A machine learning project that analyzes customer data, predicts customer churn, and provides insights to help businesses understand customer retention.

## Live Demo

[Open the Streamlit App](https://customer-churn-prediction-talha.streamlit.app/)

## Project Overview

Customer churn occurs when customers stop using a company's services. This project uses machine learning classification models to predict whether a customer is likely to leave and provides interactive dashboards for exploring churn patterns.

## Features

- Interactive Streamlit dashboard
- Customer data exploration
- Churn analysis and visualizations
- Model performance comparison
- Threshold analysis
- Individual customer churn prediction
- Retention dashboard and insights

## Machine Learning Models

The project evaluates the following classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report
- ROC curve and Precision-Recall curve

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.30% | 52.41% | 75.67% | 61.93% | 0.840 |
| K-Nearest Neighbors | 75.23% | 53.46% | 51.60% | 52.52% | 0.779 |
| Decision Tree | 76.51% | 55.01% | 63.10% | 58.78% | 0.723 |
| Random Forest | 78.28% | 61.04% | 50.27% | 55.13% | 0.816 |

*Metrics are based on the project's recorded evaluation results.*

## Dataset

The project uses the Telco Customer Churn dataset.

- Source: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Dataset size: 7,043 records
- Original dataset features: 21 columns

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

## Repository Structure

```text
cutomer-churn-prediction/
├── app.py
├── best_churn_model.pkl
├── model_features.pkl
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Talha-Arshad8/cutomer-churn-prediction.git
   ```

2. Open the project folder:

   ```bash
   cd cutomer-churn-prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Project Goals

- Understand customer churn patterns.
- Compare multiple classification models.
- Explore the factors associated with customer churn.
- Demonstrate machine learning through an interactive web application.

## Author

**Talha Arshad**

- GitHub: [Talha-Arshad8](https://github.com/Talha-Arshad8)
- LinkedIn: [Talha Arshad](https://linkedin.com/in/talha08)
