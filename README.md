# California Housing Price Prediction (Original 1990 Dataset)

This project builds a **robust machine learning pipeline** to predict **median house values** using the **original 1990 California Housing dataset** obtained from **OpenML**.  
Unlike the simplified version available in `sklearn.datasets`, this dataset includes the categorical feature **`ocean_proximity`** and reflects the **original census-level housing data**.

The project focuses on **end-to-end machine learning practices**, including exploratory data analysis (EDA), custom feature engineering, model training, hyperparameter optimization, and thorough evaluation.

---

## Dataset Information

- **Source:** OpenML (`california_housing`, version 1)
- **Year:** 1990 Census Data
- **Target Variable:** `median_house_value` (in **USD**)
- **Key Difference from sklearn dataset:**
  - Includes **categorical feature** `ocean_proximity`
  - Preserves original data distribution and feature richness

---

## Exploratory Data Analysis (EDA)

Key analyses performed:

- Distribution of target variable (train vs test)
- Feature-wise histograms
- Scatter matrix for numerical attributes
- Geographic visualization using **latitude & longitude**
- Correlation analysis (feature–feature and feature–target)
- Distribution comparison before and after **log transformations**

---

## Feature Engineering

A **custom scikit-learn–compatible transformer** (`FeatureEngineer`) was implemented to ensure clean and reusable preprocessing.

### Key steps:
- **Missing value imputation**
  - `KNNImputer` for `total_bedrooms`
- **New feature creation**
  - `income_cat` (income-based categorization)
  - `bedrooms_per_house`
- **Log transformations** for skewed features:
  - `population`
  - `total_rooms`
  - `total_bedrooms`
- **One-hot encoding** of `ocean_proximity`
- Strict separation of **training-only fitting** to prevent data leakage

---

## Model Used

### XGBoost Regressor
- Well-suited for structured/tabular data
- Captures non-linear relationships efficiently
- Trained on fully engineered feature set

---

## Hyperparameter Optimization

- **GridSearchCV** with 5-fold cross-validation
- Tuned parameters:
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`
- Evaluation metric: **RMSE**

---

## Model Performance

| Metric                             | Before Optimization | After Optimization |
|------------------------------------|---------------------|--------------------|
| Cross-Validation RMSE (Mean ± std) | 47,994 ± 613.6      | 45,797 ± 400       |
| Test RMSE                          | 48,645.4            | 45,687.4           |
| Test MAE                           | 32,001.6            | 29,585.1           |
| R² Score                           | 0.8190              | 0.8407             |


> ✅ Optimization led to **lower error**, **better generalization**, and **higher explained variance**.

---

## Key Insights

- **Median Income** is the strongest predictor of house prices, showing a **correlation of ~0.69** with the target.
- **Log transformations** significantly improved feature distributions, making them closer to Gaussian and more suitable for tree-based models.
- Geographic patterns (latitude & longitude) reveal clear regional pricing trends.
- Feature engineering contributed substantially to performance gains, even before model tuning.

---

## Model Diagnostics & Visualizations

- Learning curves (bias–variance analysis)
- Actual vs Predicted plots
- Residual analysis
- Feature importance (XGBoost gain-based)
- Permutation importance
- Confidence interval estimation for RMSE
- Geographic comparison of actual vs predicted house values

---

## Model Persistence

The final optimized model is saved using `joblib`:

The final optimized **XGBoost regression model** is serialized using **joblib** and saved as:

`California_house_prediction_xgboost_reg.pkl`

This allows the trained model to be easily reloaded later for inference or deployment without retraining, ensuring reproducibility and efficient reuse.

---

##  Author  

*Lavan Kumar Konda*  
-  Student at NIT Andhra Pradesh  
-  Passionate about Data Science, Machine Learning, and AI  
-  [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
