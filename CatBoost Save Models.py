# -*- coding: utf-8 -*-
"""
Gradient Boosting Desicion Tree Model for CDS Spread Estimation
@depencies: CatBoost and SKLearn for ML Models
@author: D Franssen
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import json

# Define paths to data and hyperparameters
data_path = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Data"
data_file = "Nomura_extended_data.xlsx"
data_file_path = os.path.join(data_path, data_file)

output_path = r"C:\Users\denni\OneDrive\Documents\Dennin\Erasmus Univeristy Rotterdam\Block 3\Master Class\Case\Output"

json_file = "optimized_hyperparameters.json"
json_file_path = os.path.join(data_path, json_file)

# Load all sheets (each sheet corresponds to a different CDS spread)
sheet_dict = pd.read_excel(data_file_path, sheet_name=None)

# Load optimized hyperparameters (obtained via TSCV and Bayesian optimization)
with open(json_file_path, "r") as f:
    optimized_hyperparameters = json.load(f)

# Define features and monotone constraints
categorical_features = ["Sector", "Region", "Country", "Ccy", 
                        "AvRating", "DocClause", "Tier"]

"""
# Define ordinal mappings
ordinal_mappings = {
    "AvRating": {'D': 8, 'CCC': 7, 'B': 6, 'BB': 5, 'BBB': 4, 'A': 3, 
                 'AA': 2, 'AAA': 1},
    "DocClause": {'XR': 8, 'XR14': 7, 'MR': 6, 'MR14': 5, 'MM': 4, 
                  'MM14': 3, 'CR': 2, 'CR14': 1},
    "Tier": {'SECDOM': 4, 'SNRFOR': 3, 'SNRLAC': 2, 'SUBLT2': 1}
}

# Define features and monotone constraints
categorical_features = ["Sector", "Region", "Country", "Ccy"]
ordinal_features = ["AvRating_ordinal", "DocClause_ordinal", "Tier_ordinal"]
monotone_constraints = [0, 0, 0, 0, 0, 0, 1, -1, -1]
"""

# Create a directory to save the final models (if it doesn't exist)
models_dir = os.path.join(output_path, "unrestricted_final_models")
os.makedirs(models_dir, exist_ok=True)

# Loop over each spread (sheet) and retrain the model on the full dataset
for spread, df in sheet_dict.items():
    print(f"Training final model for {spread}...")
    
    """
    # Apply ordinal encoding
    for col, mapping in ordinal_mappings.items():
        df[f"{col}_ordinal"] = df[col].map(mapping)
        
    # Drop original categorical, unwanted columns and the target spread column
    X = df.drop(columns=["AvRating", "DocClause", "Tier",
                         "Unnamed: 0", "Date", "Ticker", 
                         "ShortName"], errors="ignore")
    """
    # Drop original categorical and unnecessary columns
    X = df.drop(columns=["Unnamed: 0", "Date", "Ticker", 
                         "ShortName"], errors="ignore")
    
    X = X.drop(columns=[spread], errors="ignore")
    
    # Use the log transformation of the spread as the target variable
    y = np.log(df[spread])
    
    # Retrieve the tuned hyperparameters and add fixed parameters for retraining
    params = optimized_hyperparameters[spread]["Best Parameters"]
    params.update({
        "loss_function": "RMSE",
        # "monotone_constraints": monotone_constraints,
        "cat_features": categorical_features,
    })
    
    # Create a CatBoost Pool with the full dataset
    train_pool = Pool(X, label=y, cat_features=categorical_features)
    
    # Initialize and train the final CatBoost model using the full dataset
    final_model = CatBoostRegressor(**params, verbose=500)
    final_model.fit(train_pool)
    
    # Save the trained model to disk for later use
    model_filename = f"final_model_{spread}.cbm"
    model_path = os.path.join(models_dir, model_filename)
    final_model.save_model(model_path)
    
    print(f"Saved final model for {spread} to {model_path}")

print("All final models have been trained and saved.")
