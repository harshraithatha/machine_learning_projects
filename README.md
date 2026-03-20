# Machine Learning Projects Portfolio

This repository is a simple, interview-friendly machine learning portfolio built around classic supervised ML workflows. It focuses on regression, classification, time-aware forecasting features, ensemble models, and practical model packaging.

## Projects

1. `01_house_price_regression`
   Synthetic housing-style regression using Linear Regression, Decision Tree, Random Forest, and XGBoost.
2. `02_customer_churn_classification`
   Binary classification on telecom-style churn data using Logistic Regression, Decision Tree, Random Forest, and XGBoost.
3. `03_demand_forecasting`
   Retail demand forecasting with lag features, calendar features, and regression models.
4. `04_fraud_detection`
   Imbalanced transaction classification using tree models and boosted models.
5. `05_sales_prediction_app`
   A small Streamlit app backed by a trained regression model for sales prediction.

## What This Covers

- Regression and classification
- Decision Trees and Random Forests
- XGBoost for both regression and classification
- Data preprocessing for numeric and categorical features
- Model comparison and metric tracking
- Feature engineering for forecasting
- Handling imbalanced classification problems
- Saving trained models and using them for inference

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate local datasets:

```bash
python scripts/generate_portfolio_data.py
```

Train all projects:

```bash
python run_all.py
```

Run the Streamlit app:

```bash
streamlit run projects/05_sales_prediction_app/app.py
```
