# Car Price Prediction

A supervised machine learning project completed as part of the **MSc Artificial Intelligence & Data Science** programme at the **University of Hull**.

The project builds and evaluates regression models to predict used car prices based on vehicle attributes.

---

## Overview

Using a car sales dataset (`car_sales_data.csv`), this project covers end-to-end ML workflow — from exploratory data analysis and feature engineering through to model training, evaluation, and comparison across multiple regression approaches.

---

## Project Structure

```
├── Car_project.ipynb       # Main notebook
├── car_sales_data.csv      # Dataset used for training and evaluation
└── README.md
```

---

## What the Notebook Covers

### 1. Data Loading & Inspection
- Loaded car sales data using `pandas`
- Inspected column types, null values, and unique categorical values
- Explored distributions with `seaborn` histograms (e.g. Engine size)

### 2. Feature Selection
| Type | Features |
|------|----------|
| Numeric | Engine size, Year of manufacture, Mileage |
| Categorical | Manufacturer, Model, Fuel type |
| Target | Price |

### 3. Train / Test Split
- 80/20 train-test split using `train_test_split` with `random_state=42`

### 4. Models Built

#### Single-Feature Regression (per numeric feature)
For each of the three numeric features, two models were trained and compared:

- **Linear Regression** — with `StandardScaler` normalisation
- **Polynomial Regression (degree 2)** — built on top of scaled features using `PolynomialFeatures`

#### Evaluation Metrics
A reusable `regression_metrics()` function was implemented to compute:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |

All single-feature model results were collected into a summary DataFrame and ranked by RMSE.

---

## Tools & Libraries

| Library | Usage |
|---------|-------|
| `pandas` | Data loading and manipulation |
| `numpy` | Array operations |
| `seaborn` / `matplotlib` | EDA visualisations |
| `scikit-learn` | Preprocessing, modelling, evaluation |

---

## Key Concepts Demonstrated

- Supervised learning workflow (train → scale → fit → predict → evaluate)
- Feature standardisation with `StandardScaler`
- Polynomial feature expansion for non-linear relationships
- Model comparison using regression metrics
- Clean, reusable metric reporting functions

---

## Author

**Azeez Abdul-Majeed Babatunde**  
MSc Artificial Intelligence & Data Science  
University of Hull
