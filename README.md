# Car Price Prediction — Regression Modelling on Used Car Sales Data

---

## What this project does

This notebook builds and compares regression models to predict used car prices from vehicle attributes. The work covers the full ML workflow: data inspection, EDA, feature selection, preprocessing, model training, and evaluation.

The focus was on understanding *which features actually drive price* and *whether adding polynomial terms improves over a simple linear baseline* — rather than reaching for complex models unnecessarily.

---

## Dataset

**File:** `car_sales_data.csv`

Key features used:

| Type | Features |
|------|----------|
| Numeric | Engine size, Year of manufacture, Mileage |
| Categorical | Manufacturer, Model, Fuel type |
| Target | Price |

Pre-processing steps applied:
- Checked for missing values across all columns
- Explored feature distributions with histograms (Engine size)
- Inspected unique values for categorical columns (e.g. Fuel type)
- 80/20 train-test split with `random_state=42`
- `StandardScaler` applied to numeric features before model training

---

## Models implemented

For each of the three numeric features, two models were trained and compared in isolation (single-feature approach):

**Linear Regression**
- Feature scaled with `StandardScaler`
- Fitted using `sklearn.linear_model.LinearRegression`

**Polynomial Regression (degree 2)**
- Built on top of scaled features using `PolynomialFeatures(degree=2)`
- Same `LinearRegression` estimator applied to the expanded feature set

A reusable `regression_metrics()` function was implemented to keep evaluation consistent across all models:

```python
def regression_metrics(y_true, y_pred, model_name=None):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    ...
```

All results were collected into a summary DataFrame and ranked by RMSE.

**Metrics tracked:**

| Metric | What it tells you |
|--------|------------------|
| MAE | Average absolute error in price prediction |
| RMSE | Penalises large errors more heavily than MAE |
| R² | Proportion of price variance explained by the model |

---

## Key findings

Single-feature models using **Year of manufacture** and **Engine size** tended to outperform Mileage alone — newer cars and larger engines correlate more directly with price in this dataset.

Polynomial regression (degree 2) improved on linear for most features, capturing the non-linear relationship between age/mileage and price depreciation. However, the gains were modest, suggesting the relationship is partially linear and that multi-feature models would be the natural next step.

The gradient and intercept of the final linear model were extracted and printed, providing interpretability: the coefficient tells you how much price changes per unit shift in the scaled feature.

---

## How to run

```bash
# Clone the repo
git clone https://github.com/Meezxbt2001/car-price-prediction.git
cd car-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Car_project.ipynb
```

Make sure `car_sales_data.csv` is in the same directory as the notebook before running.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Project structure

```
car-price-prediction/
│
├── Car_project.ipynb       # Main notebook
├── car_sales_data.csv      # Dataset
├── requirements.txt
└── README.md
```

---

## What's next

This project used single-feature regression as a deliberate starting point. Natural extensions include:

- Multi-feature regression using all numeric + one-hot encoded categorical features
- Ridge / Lasso regularisation to handle multicollinearity
- Tree-based models (Random Forest, XGBoost) for comparison
- Feature importance analysis to identify the strongest price predictors

---

## References

- Pedregosa, F. et al. (2011) Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, pp. 2825–2830.
- James, G. et al. (2021) *An Introduction to Statistical Learning*. 2nd edn. Springer.
- Bishop, C.M. (2006) *Pattern Recognition and Machine Learning*. Springer.

---

## Author

**Azeez Abdul-Majeed Babatunde**
MSc Artificial Intelligence & Data Science
University of Hull
