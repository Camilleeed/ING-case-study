#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:37:42 2025

@author: lufan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:20:10 2025

@author: lufan
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

file_path = "/Users/lufan/Documents/erasmus_classes/master blok 3/seminar/data_cleaning/Nomura_extended_data.xlsx"

xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names 

all_results = {}

for sheet in sheet_names:
    print(f"Processing sheet: {sheet}...")  # Progress tracker
    data = pd.read_excel(file_path, sheet_name=sheet)    
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S')
    results_list = []

    # Loop through each unique date
    for date_of_interest in data['Date'].unique():
        print(f"  Processing Date: {date_of_interest.date()}")  # Progress update
        group = data[data['Date'] == date_of_interest]

        # Create dummy variables
        group = pd.get_dummies(group, columns=['Sector', 'Region', 'AvRating', 'Tier', 'DocClause', 'Ccy'], dtype=int)

        # List of reference categories
        reference_categories = {
            'Sector_Healthcare',
            'Region_North America',
            'AvRating_BBB',
            'Tier_SNRFOR',
            'DocClause_MR14',
            'Ccy_JPY'
        }

        # Create interaction terms between Region and AvRating
        for region_col in [col for col in group.columns if col.startswith('Region_')]:
            for rating_col in [col for col in group.columns if col.startswith('AvRating_')]:
                interaction_col = f"{region_col}_x_{rating_col}"
                group[interaction_col] = group[region_col] * group[rating_col]
                
    
        # Drop reference categories to avoid multicollinearity
        group = group.drop(columns=reference_categories, errors='ignore')

        # Drop only the redundant interaction terms that cause multicollinearity
        group = group.drop(columns=['AvRating_BBB_x_Region_North America'], errors='ignore')

        # Handle log transformation safely
        group['LogSpread'] = np.log(group[sheet].replace(0, np.nan))  # Replace zero with NaN before log
        group = group.dropna(subset=['LogSpread'])  # Remove rows where log transformation failed

        # Prepare y
        y = group['LogSpread']
        
        # Prepare X
        X = group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0', 'CompositeDepth5y'], axis=1, errors='ignore')

        # Add constant term
        X.insert(0, 'Constant', 1)

        # Skip regression if X is singular (not enough variation)
        if X.shape[1] <= 1:  # If there's only the constant term
            print(f"  Skipping Date: {date_of_interest.date()} due to insufficient predictors")
            continue

        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Compute MSE and adjusted R^2
        y_hat = model.predict(X)
        mse = mean_squared_error(y, y_hat)
        R2 = model.rsquared_adj

        # Store the coefficients and MSE for this date
        result = {'Date': date_of_interest, 'MSE': mse, 'adjusted R2': R2}
        for coef_name, coef_value in model.params.items():
            result[coef_name] = coef_value
        
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    all_results[sheet] = results_df

# Save results to an Excel file
output_file = "/Users/lufan/Documents/erasmus_classes/master blok 3/seminar/data_cleaning/extended_model_1_regression_results.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in all_results.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Results saved to {output_file}")
