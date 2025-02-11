# -*- coding: utf-8 -*-
"""
Gradient Boosting Desicion Tree Model for CDS Spread Estimation
@depencies: CatBoost, Optuna (Bayesian HPO) and SKLearn (ML INFRA)
@author: D Franssen
"""

import pandas as pd
import numpy as np
import optuna
import optuna_dashboard
from catboost import CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, KFold
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


## Hyperparamter Tuning (Optuna)


# Define SQLite storage
storage = "sqlite:///db.sqlite3"

# Dictionary to store best parameters & results
optimized_results = {}

# Keep a copy of the full feature set
original_X, original_y = X.copy(), spreads.copy()

# Loop through each spread maturity and optimize
for spread in ["Spread6m", "Spread5y"]:
    # "Spread6m", "Spread5y",
    # "Spread1y", "Spread2y", "Spread3y", "Spread4y"
    # "Spread15y", "Spread30y"
    # "Spread7y", "Spread10y", "Spread20y"
    print(f"\n🔹 Optimizing CatBoost for {spread}...\n")

    # Reset X and y for each spread to avoid accumulating dropped values
    X, y = original_X.copy(), original_y[spread].copy()
    
    # Apply log transformation to target
    y = np.log(spreads[spread])

    # Drop NaN values
    X, y = X[~y.isna()], y.dropna()
    
    # Save Optuna Study
    study_name = f"optuna-{spread}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded existing study: {study_name}")
    except KeyError:
        study = optuna.create_study(storage=storage, direction='minimize')
        print(f"📄 Created new study: {study_name}")
        
    # Define Optuna optimization function
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'boosting_type': 'Ordered',
            'min_data_in_leaf': 100,
            'loss_function': 'RMSE'
        }

        # Create CatBoost Pool
        full_pool = Pool(X, label=y, cat_features=categorical_features)  
        
        # Perform 5-Fold Cross-Validation using CatBoost
        cv_results = cv(
            full_pool,
            params,
            fold_count=5,  # 5-fold CV
            partition_random_seed=42,  # Ensures reproducibility
            shuffle=True,
            stratified=False, 
            verbose=250
        )
        
        return np.min(cv_results['test-RMSE-mean']) 

    # Run Optuna optimization with CV
    study.optimize(objective, n_trials=1)

    # Get best model parameters
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best Parameters for {spread}: {best_params}")
    print(f"Best Score (MSE): {best_score}")

    # Train the best model
    best_model = CatBoostRegressor(**best_params, loss_function='RMSE', cat_features=categorical_features, verbose=100)
    best_model.fit(X, y, early_stopping_rounds=50)

    # Make predictions
    y_pred = best_model.predict(X)

    # Store results
    optimized_results[spread] = {
        "Best Parameters": best_params,
        "Best MSE": best_score,
        "R^2 Score": r2_score(y, y_pred)
    }

print(f"{optimized_results}")

# Save optimized results to an Excel file
# output_file = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data\Optimized_CatBoost_Results.xlsx"
# pd.DataFrame.from_dict(optimized_results, orient="index").to_excel(output_file)
# print(f"\n Optimized results saved successfully: {output_file}")

# Launch the dashboard
optuna_dashboard.run_server(storage="sqlite:///db.sqlite3", host="127.0.0.1", port=8080)


## Cross-fold Validation ##


# Dictionary to store final cross-validation results
cv_results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for spread, best_params in optimized_results.items():
    print(f"\n🔹 Running 5-Fold Cross-Validation for {spread}...\n")
    
    # Reset X and y for each spread to avoid accumulating dropped indices
    X, y = original_X.copy(), original_y[spread].copy()
    
    # Apply log transformation to target
    y = np.log(y)  # log to keep consistency with previous training

    # Drop NaNs while ensuring X and y have matching indices
    valid_indices = y.dropna().index  # Get valid indices after dropping NaNs from y
    X, y = X.loc[valid_indices], y.loc[valid_indices]  # Keep only matching rows

    # Define CatBoost parameters (use best parameters from Optuna tuning)
    params = best_params["Best Parameters"]
    params.update({
        "loss_function": "RMSE",  
        "cat_features": categorical_features,
        "verbose": 0
    })

    fold_preds, fold_actuals = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=categorical_features, verbose=500)

        y_pred = model.predict(X_test)

        fold_preds.extend(y_pred)
        fold_actuals.extend(y_test)

    # Compute MSE and R²
    final_mse = mean_squared_error(fold_actuals, fold_preds)
    final_r2 = r2_score(fold_actuals, fold_preds)
    
    # Store CV results
    cv_results[spread] = {
        "Final CV MSE": final_mse,
        "Final CV R²": final_r2,
        "Best Parameters": best_params["Best Parameters"]
        }
    print(f"{spread} - Final CV RMSE: {final_mse:.6f}, R²: {final_r2:.6f}")

print(f"{cv_results}")
