# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:06:38 2025

@author: denni
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error

def apply_ordinal_mappings(df, ordinal_mappings):
    """
    1) Creates new columns, e.g. 'AvRating_ordinal', from original columns.
    2) Drops rows that have missing ordinal values (NaN) in these new columns.
    3) Returns a *copy* of the cleaned DataFrame.
    """
    df = df.copy()  # avoid mutating the original
    for col, mapping in ordinal_mappings.items():
        df[f"{col}_ordinal"] = df[col].map(mapping)
    
    # Drop rows that have NaN in *any* of the ordinal columns
    ordinal_cols = [f"{col}_ordinal" for col in ordinal_mappings.keys()]
    df.dropna(subset=ordinal_cols, inplace=True)
    return df

# ---------------------------
# 1. Setup Paths and Filenames
# ---------------------------

data_path = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data"
output_path = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Output"

# Files
cleaned_oos_file = "cleaned_oos.xlsx"  # Out-of-sample file WITHOUT spreads (y)
oos_with_y_file = "markit_out_of_sample_202302_202303.xlsx"  # Out-of-sample file WITH spreads (used for evaluation)

# Model directories
restricted_models_dir = os.path.join(output_path, "restricted_final_models")
unrestricted_models_dir = os.path.join(output_path, "unrestricted_final_models")

# ---------------------------
# 2. Define Model Settings and Column Lists
# ---------------------------

# Spread (target) columns
spread_columns = [
    "Spread6m", "Spread1y", "Spread2y", "Spread3y", "Spread4y", 
    "Spread5y", "Spread7y", "Spread10y", "Spread15y", "Spread20y", "Spread30y"
]

# For restricted models:
# (Some are replaced by ordinal mappings)
drop_columns_restricted = [
    "ImpliedRating", "AvRating", "DocClause", "Tier", "Unnamed: 0", "Date", 
    "Ticker", "ShortName"
]

# For unrestricted models:
drop_columns_unrestricted = [
    "ImpliedRating", "Unnamed: 0", "Date", "Ticker", "ShortName"
]

# Categorical features
cat_features_unrestricted = [
"Sector", "Region", "Country", "Ccy", "AvRating", "DocClause", "Tier"
]

cat_features_restricted   = ["Sector", "Region", "Country", "Ccy"]

# Define ordinal mappings (as used in training for restricted models)
ordinal_mappings = {
    "AvRating": {'D': 8, 'CCC': 7, 'B': 6, 'BB': 5, 'BBB': 4, 'A': 3, 'AA': 2, 'AAA': 1},
    "DocClause": {'XR': 8, 'XR14': 7, 'MR': 6, 'MR14': 5, 'MM': 4, 'MM14': 3, 'CR': 2, 'CR14': 1},
    "Tier": {'SECDOM': 4, 'SNRFOR': 3, 'SNRLAC': 2, 'SUBLT2': 1}
}

# ---------------------------
# 3. Load Data and Pre-filter
# ---------------------------

# Load the cleaned OOS file (without y)
df_clean = pd.read_excel(os.path.join(data_path, cleaned_oos_file))

# Remove rows that do not have AvRating (applies for both models)
df_clean = df_clean.dropna(subset=["AvRating"]).copy()

# Load the OOS file with spreads (used for evaluation)
df_oos_with_y = pd.read_excel(os.path.join(data_path, oos_with_y_file))

# ---------------------------
# 4. Prepare for Final Predictions and Evaluation Metrics
# ---------------------------

# We will create two copies of the cleaned file
df_pred_log_res = df_clean.copy()
df_pred_spread_res = df_clean.copy()
df_pred_log_unres = df_clean.copy()
df_pred_spread_unres = df_clean.copy()

eval_metrics = []  # to store per-maturity evaluation results

# ---------------------------
# 5. Loop Over Each Spread and Predict
# ---------------------------

for spread in spread_columns:
    # Determine rows that can be checked, where df_oos_with_y has a spread.
    valid_idx = df_oos_with_y.dropna(subset=[spread]).index.intersection(df_clean.index)
    if valid_idx.empty:
        print(f"No valid rows for spread {spread}. Skipping.")
        continue
    
    # Actual y (in log) for evaluation: 
    # Note that models were trained on log(spread)
    y_true_log = np.log(df_oos_with_y.loc[valid_idx, spread])

    # Actual y for evaluation: 
    y_true = df_oos_with_y.loc[valid_idx, spread]

    # ---------------------------
    # 5.1 Restricted Model Prediction
    # ---------------------------
    
    df_temp_res = df_clean.loc[valid_idx].copy()
    df_temp_res = apply_ordinal_mappings(df_temp_res, ordinal_mappings)
    
    # Prepare features: drop the unwanted columns and all spread columns.
    X_res = df_temp_res.drop(columns=drop_columns_restricted + spread_columns, errors="ignore")
    
    # Load the restricted model for the current spread.
    res_model_path = os.path.join(restricted_models_dir, f"final_model_{spread}.cbm")
    if not os.path.exists(res_model_path):
        print(f"Restricted model for {spread} not found. Skipping.")
        continue
    
    model_res = CatBoostRegressor()
    model_res.load_model(res_model_path)

    # Predict log spread
    y_pred_log_res = model_res.predict(X_res)
    # Take exponential to get the normal spread
    y_pred_spread_res = np.exp(y_pred_log_res)

    # Compute RMSE on log scale
    rmse_log_res = root_mean_squared_error(y_true_log ,y_pred_log_res)
    
    # Compute RMSE on normal scale
    rmse_spread_res = root_mean_squared_error(y_true, y_pred_spread_res)

    # ---------------------------
    # 5.2 Unrestricted Model Prediction
    # ---------------------------

    df_temp_unres = df_clean.loc[valid_idx].copy()

    # For unrestricted, drop  missing values in categorical features
    for col in cat_features_unrestricted:
        if col in df_temp_unres.columns:
            df_temp_unres.dropna(subset=[col])

    # Prepare features by dropping the unwanted columns and spread columns.
    X_unres = df_temp_unres.drop(columns=drop_columns_unrestricted + spread_columns, errors="ignore")

    # Load the unrestricted model for the current spread.
    unres_model_path = os.path.join(unrestricted_models_dir, f"final_model_{spread}.cbm")
    if not os.path.exists(unres_model_path):
        print(f"Unrestricted model for {spread} not found. Skipping.")
        continue

    model_unres = CatBoostRegressor()
    model_unres.load_model(unres_model_path)
    
    # Predict log spread
    y_pred_log_unres = model_unres.predict(X_unres)
    y_pred_spread_unres = np.exp(y_pred_log_unres)
    
    # Compute RMSE on log scale
    rmse_log_unres = root_mean_squared_error(y_true_log ,y_pred_log_unres)
    
    # Compute RMSE on normal scale
    rmse_spread_unres = root_mean_squared_error(y_true, y_pred_spread_unres)
    
    # The number of predicted rows should be the same (use the valid index from df_oos_with_y)
    n_rows = len(valid_idx)
    
    # ---------------------------
    # 5.3 Save Evaluations and Predictions
    # ---------------------------
    
    # Append evaluation results for this spread
    eval_metrics.append({
        "Spread": spread,
        "NumPredictedRows": n_rows,
        "RMSE_restricted_log": rmse_log_res,
        "RMSE_restricted_spread": rmse_spread_res,
        "RMSE_unrestricted_log": rmse_log_unres,
        "RMSE_unrestricted_spread": rmse_spread_unres,
    })
    
    # For restricted model predictions
    df_pred_log_res.loc[valid_idx, spread] = y_pred_log_res
    df_pred_log_res.rename(columns={spread: f"{spread}_log_pred"}, inplace=True)

    df_pred_spread_res.loc[valid_idx, spread] = y_pred_spread_res
    df_pred_spread_res.rename(columns={spread: f"{spread}_spread_pred"}, inplace=True)

    # For unrestricted model predictions:
    df_pred_log_unres.loc[valid_idx, spread] = y_pred_log_unres
    df_pred_log_unres.rename(columns={spread: f"{spread}_log_pred"}, inplace=True)

    df_pred_spread_unres.loc[valid_idx, spread] = y_pred_spread_unres
    df_pred_spread_unres.rename(columns={spread: f"{spread}_spread_pred"}, inplace=True)

# ---------------------------
# 6. Save Evaluation Results to Excel
# ---------------------------

eval_df = pd.DataFrame(eval_metrics)
eval_results_file = os.path.join(output_path, "catboost_nomissing_oos_evaluation.xlsx")
eval_df.to_excel(eval_results_file, index=False)
print("Evaluation results saved to:", eval_results_file)

# ---------------------------
# 7. Save Final Predictions (with actual outputs) to One Excel File with Two Sheets
# ---------------------------

# Here each sheet contains the original cleaned file plus new prediction columns (log-spreads) for each maturity.
predictions_log_file = os.path.join(output_path, "catboost_nomissing_oos_predictions_log.xlsx")
with pd.ExcelWriter(predictions_log_file) as writer:
    df_pred_log_res.to_excel(writer, sheet_name="restricted", index=False)
    df_pred_log_unres.to_excel(writer, sheet_name="unrestricted", index=False)
print("Predicted outputs saved in:", predictions_log_file)


# Here each sheet contains the original cleaned file plus new prediction columns (spreads) for each maturity.
predictions_spread_file = os.path.join(output_path, "catboost_nomissing_oos_predictions_spread.xlsx")
with pd.ExcelWriter(predictions_spread_file) as writer:
    df_pred_spread_res.to_excel(writer, sheet_name="restricted", index=False)
    df_pred_spread_unres.to_excel(writer, sheet_name="unrestricted", index=False)
print("Predicted outputs saved in:", predictions_spread_file)