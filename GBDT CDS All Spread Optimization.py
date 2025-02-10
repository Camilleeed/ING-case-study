# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:19:30 2025

@author: denni
"""

import pandas as pd
import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data\markit_in_sample_202302_202303.xlsx"
df = pd.read_excel(file)
copy_df = df.copy()

# Define categorical features
categorical_features = ["Sector", "Region", "AvRating", "Tier", "Country", "DocClause", "Ccy"]
for col in categorical_features:
    df[col] = df[col].fillna('nan').astype(str)  # Convert NaN in categorical features to string

# Drop unnecessary columns & extract spreads
df = df.drop(columns=["Column1", "Ticker", "ShortName"])
spreads = df[["Date", "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", "Spread5y",
              "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y"]]
X = df.drop(columns=["Date", "ImpliedRating", "Recovery", "CompositeDepth5y", "CurveLiquidityScore"] + list(spreads.columns))

# Dictionary to store best parameters & results
optimized_results = {}

# Loop through each spread maturity and optimize separately
for spread in ["Spread7y", "Spread10y", "Spread20y"]: 
    # "Spread6m", "Spread5y",
    # "Spread1y", "Spread2y", "Spread3y", "Spread4y"
    # "Spread15y", "Spread30y"
    print(f"\n🔹 Optimizing CatBoost for {spread}...\n")

    # Apply log transformation to target
    y = np.log(spreads[spread])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Drop NaN target values
    X_train, y_train = X_train[~y_train.isna()], y_train.dropna()
    X_test, y_test = X_test[~y_test.isna()], y_test.dropna()

    # Create CatBoost Pool
    train_pool = Pool(X_train, label=y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, label=y_test, cat_features=categorical_features)

    # Define Optuna optimization function
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'boosting_type': 'Ordered',
            'min_data_in_leaf': 100
        }

        # Train model
        model = CatBoostRegressor(**params, loss_function='RMSE', cat_features=categorical_features, verbose=100)
        model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=100)

        # Evaluate model
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse
    
    # Run Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Get best model parameters
    best_params = study.best_params
    best_score = study.best_value
    print(f"✅ Best Parameters for {spread}: {best_params}")
    print(f"✅ Best Score (MSE): {best_score}")

    # Train the best model
    best_model = CatBoostRegressor(**best_params, loss_function='RMSE', cat_features=categorical_features, verbose=100)
    best_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50, verbose=100)

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Store results
    optimized_results[spread] = {
        "Best Parameters": best_params,
        "Best MSE": best_score,
        "R^2 Score": r2_score(y_test, y_pred)
    }
    
print(f"{optimized_results}")

# Save optimized results to an Excel file
output_file = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data\Optimized_CatBoost_Results.xlsx"
pd.DataFrame.from_dict(optimized_results, orient="index").to_excel(output_file)
print(f"\n✅ Optimized results saved successfully: {output_file}")