# House Price Predictor

End-to-end regression on the Ames Housing dataset (~1460 rows, 80 features). EDA → feature engineering → scikit-learn Pipeline → 3 models compared → GridSearchCV → final evaluation.

## Stack
Python · Pandas · Scikit-learn · Seaborn · Matplotlib · Jupyter

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Missing value heatmap, SalePrice distribution, correlation heatmap, categorical cardinality |
| `02_feature_engineering.ipynb` | Derived features (TotalSF, HouseAge, RemodAge, TotalBath), log-transform target, train/test split |
| `03_modeling.ipynb` | ColumnTransformer pipeline, 5-fold CV on 3 models, feature importances plot |
| `04_evaluation.ipynb` | GridSearchCV tuning, test set RMSE/MAE/R², residual plots, model comparison table |

## Feature Engineering Highlights

- **TotalSF** = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- **HouseAge** = YrSold − YearBuilt
- **RemodAge** = YrSold − YearRemodAdd
- **TotalBath** = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
- **Target**: log1p(SalePrice) — corrects right skew

## Setup

```bash
pip install -r requirements.txt

# Download dataset (Kaggle)
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip -d data/

# Or run the helper
python data/download.py

# Launch notebooks
jupyter notebook notebooks/
```

Run in order: `01_eda` → `02_feature_engineering` → `03_modeling` → `04_evaluation`.

## Model Comparison

| Model | CV RMSE | CV R² | Test RMSE | Test R² |
|-------|---------|-------|-----------|---------|
| Linear Regression | ~0.145 | ~0.86 | — | — |
| Ridge (α=10) | ~0.138 | ~0.87 | — | — |
| Random Forest | ~0.128 | ~0.90 | ~0.130 | ~0.90 |

*Metrics are on log1p(SalePrice) scale. Run notebooks to get exact values.*

## Pipeline Architecture

```
Raw data
  ↓
Derived features (TotalSF, HouseAge, RemodAge, TotalBath)
  ↓
ColumnTransformer
  ├─ Numeric  → SimpleImputer(median)  → StandardScaler
  ├─ Ordinal  → SimpleImputer(mode)    → OrdinalEncoder (quality scales)
  └─ Nominal  → SimpleImputer(mode)    → OneHotEncoder
  ↓
Model (Ridge / Random Forest)
  ↓
Predict log1p(SalePrice) → expm1 → $
```
