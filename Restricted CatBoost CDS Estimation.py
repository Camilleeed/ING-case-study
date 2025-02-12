# -*- coding: utf-8 -*-
"""
(Restriced) Gradient Boosting Desicion Tree Model for CDS Spread Estimation.
    Uses TimeSeries Cross(5)-Fold Validation to learn only from past data,
    with ordinal mappings for AvRating, DocClause and Tier enforced by
    monotone const
    
@depencies: Pandas, 
            Numpy, 
            CatBoost, and 
            SKLearn (Model Selection: TimeSeries CV. 
                     Metrics: MSE and R2.)
@author: D Franssen
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

path = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data"
file = "Nomura_extended_data.xlsx"
file_path = f"{path}\\{file}"

sheet_dict = pd.read_excel(file_path, sheet_name=None)

# Define ordinal mappings
ordinal_mappings = {
    "AvRating": {'D': 8, 'CCC': 7, 'B': 6, 'BB': 5, 'BBB': 4, 'A': 3, 'AA': 2, 'AAA': 1},
    "DocClause": {'XR': 8, 'XR14': 7, 'MR': 6, 'MR14': 5, 'MM': 4, 'MM14': 3, 'CR': 2, 'CR14': 1},
    "Tier": {'SECDOM': 4, 'SNRFOR': 3, 'SNRLAC': 2, 'SUBLT2': 1}
}

# Define features and monotone constraints
categorical_features = ["Sector", "Region", "Country", "Ccy"]
ordinal_features = ["AvRating_ordinal", "DocClause_ordinal", "Tier_ordinal"]
monotone_constraints = [0, 0, 0, 0, 0, 0, 1, -1, -1]

# Define Time Series K(=5)Fold Split
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Save cv results
cv_results = []

print(X.columns())

for spread, df in sheet_dict.items():
    print(f"Processing {spread}...")
    
    # Apply ordinal encoding
    for col, mapping in ordinal_mappings.items():
        df[f"{col}_ordinal"] = df[col].map(mapping)
    
    # Drop original categorical and unnecessary columns
    X = df.drop(columns=["AvRating", "DocClause", "Tier",
                         "Unnamed: 0", "Date", "Ticker", 
                         "ShortName"], errors="ignore")
    
    # Drop spread columns
    X = X.drop(columns=[spread], errors="ignore")
    
    # Define target y & apply log transformation
    y = np.log(df[spread])
    
    # Define CatBoost parameters (uses best parameters from Optuna tuning)
    params = optimized_results[spread]["Best Parameters"]
    params.update({
        "loss_function": "RMSE",
        "monotone_constraints": monotone_constraints,
        "cat_features": categorical_features,
    })
    
    mse_scores = []
    r2_scores = []
    
    # Perform TimeSeries Cross-Validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits} for {spread}...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Define CatBoost Pool
        train_pool = Pool(X_train, 
                          label=y_train, 
                          cat_features=categorical_features)
        test_pool = Pool(X_test, 
                         label=y_test, 
                         cat_features=categorical_features)
        
        # Initialize and Train CatBoost Model
        model = CatBoostRegressor(**params, verbose=500)
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Store Per-Fold Metrics
        cv_results.append({
            "Spread": spread,
            "Fold": fold + 1,
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
            "Best Iteration": model.get_best_iteration(),
            "Feature Importance": model.get_feature_importance().tolist()
        })

print("Cross-validation completed for all spreads!")

# Convert results to DataFrame and save
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_excel("cross_validation_results.xlsx", index=False)
print("Results saved to 'cross_validation_results.xlsx'")
