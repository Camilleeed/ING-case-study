# -*- coding: utf-8 -*-
"""
Gradient Boosting Desicion Tree Model for CDS Spread Estimation
@depencies: CatBoost, Optuna (Bayesian HPO) and SKLearn (ML INFRA)
@author: D Franssen
"""

import pandas as pd
import numpy as np
# import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

file = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data\markit_in_sample_202302_202303.xlsx"
output_file = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data\CatBoost_Spread5y.xlsx"

# Load dataset and define categorical features
df_ordinal = pd.read_excel(file)
df_ordinal_copy = df_ordinal.copy()

## Data Prep ##

ordinal_mapping_AvRating = {
    'D': 8,
    'CCC': 7,
    'B': 6,
    'BB': 5,
    'BBB': 4,
    'A': 3,
    'AA': 2,
    'AAA': 1
}

ordinal_mapping_DocClause = {
    'XR': 8,
    'XR14': 7,
    'MR': 6,
    'MR14': 5,
    'MM': 4,
    'MM14': 3,
    'CR': 2,
    'CR14': 1
}

ordinal_mapping_Tier = {
    'SECDOM': 4,
    'SNRFOR': 3,
    'SNRLAC': 2,
    'SUBLT2': 1
}

# Apply the mapping to the 'avgrating' column
df_ordinal['AvRating_ordinal'] = df_ordinal['AvRating'].map(ordinal_mapping_AvRating)
df_ordinal['DocClause_ordinal'] = df_ordinal['DocClause'].map(ordinal_mapping_DocClause)
df_ordinal['Tier_ordinal'] = df_ordinal['Tier'].map(ordinal_mapping_Tier)

categorical_features = ["Sector", "Region", "AvRating_ordinal", "Tier_ordinal", "Country", "DocClause_ordinal", "Ccy"]
category_features = ["Sector", "Region", "Country", "Ccy"]
ordinal_features =["AvRating_ordinal", "Tier_ordinal", "DocClause_ordinal"]

# Replace NaN values in categorical features with the string 'nan' and ensure all are strings
for col in category_features:
    df_ordinal[col] = df_ordinal[col].fillna('nan').astype(str)

df_ordinal = df_ordinal.dropna(subset=['AvRating_ordinal','DocClause_ordinal', 'Tier_ordinal'])

tickers = df_ordinal.copy()

# Define features (keeping all columns except the target)
df_ordinal = df_ordinal.drop(columns=["Column1", "Ticker", "ShortName", "AvRating", "DocClause", "Tier"])
leakage = df_ordinal[["Date", "ImpliedRating", "Recovery", "CompositeDepth5y", "CurveLiquidityScore"]]
spreads = df_ordinal[["Date", "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", "Spread5y", "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y"]]

X = df_ordinal.drop(columns=["ImpliedRating", "Recovery", "CompositeDepth5y", "CurveLiquidityScore", "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", "Spread5y", "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y"])

# Apply log transformation (log1p to avoid issues with zero values)
y = np.log(spreads["Spread6m"])

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove rows where target (Spread5y) is NaN
X_train = X_train[~y_train.isna()]
y_train = y_train.dropna()

X_test = X_test[~y_test.isna()]
y_test = y_test.dropna()

X_test_with_date = X_test.copy()  # Copy X_test before removing Date
X_test_with_date["Date"] = df_ordinal.loc[X.index, "Date"]  # Restore Date column

X_train = X_train.drop(columns=["Date"])
X_test = X_test.drop(columns=["Date"])

## CatBoost ##

# Define CatBoost Pool (optimized for categorical features)
train_pool = Pool(X_train, 
                  label=y_train, 
                  cat_features=category_features)

test_pool = Pool(X_test, 
                 label=y_test, 
                 cat_features=category_features)

# Initialize and Train CatBoost Model
model = CatBoostRegressor(
    depth=9,
    learning_rate=0.18564647534503254,
    iterations=500, #2234
    l2_leaf_reg=7.231674504247666,
    subsample=0.7914624753637889,
    loss_function='RMSE', # Root Mean Square Error (no MSE)
    boosting_type='Ordered',
    min_data_in_leaf=100,
    monotone_constraints=[0, 0, 0, 0, 1, -1, -1],
    cat_features=category_features,
    verbose=250  # Show progress every 250 iterations
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=True)

# Predictions
y_pred = model.predict(X_test)

## RESULTS ##

# R-Squared
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.6f}")

# Define categorical features (excluding AvRating)
categorical_features_fixed = ["Sector", "Region", "Tier", "Country", "DocClause", "Ccy"]

# Select the most frequent value (mode) for each categorical feature
fixed_values = {col: X_test_with_date[col].mode()[0] for col in categorical_features_fixed}

# Results dataframe with MSE
results_df = pd.DataFrame({
    "Date": X_test_with_date["Date"],
    "Ticker": tickers.loc[X_test_with_date.index, "Ticker"],
    "Sector": tickers.loc[X_test_with_date.index, "Sector"],
    "Region": tickers.loc[X_test_with_date.index, "Region"],
    "AvRating": tickers.loc[X_test_with_date.index, "AvRating"],
    "Tier": tickers.loc[X_test_with_date.index, "Tier"],
    "Country": tickers.loc[X_test_with_date.index, "Country"],
    "DocClause": tickers.loc[X_test_with_date.index, "DocClause"],
    "Ccy": tickers.loc[X_test_with_date.index, "Ccy"],

    "Actual Spread5y": y_test,
    "Predicted Spread5y": y_pred,
    "MSE": (y_test - y_pred) ** 2
}) 

# Save to Excel
results_df.to_excel(output_file, index=False)
print(f"File saved successfully: {output_file}")
