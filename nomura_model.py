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
    data = pd.read_excel(file_path, sheet_name=sheet)    
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y %H:%M:%S')
    results_list = []

    # Loop each date
    for date_of_interest in data['Date'].unique():
        group = data[data['Date'] == date_of_interest]

        # Create dummy variables
        group = pd.get_dummies(group, columns=['Sector', 'Region', 'AvRating', 'Tier'], dtype=int)

        # List of reference categories
        reference_categories = {
            'Sector_Healthcare',
            'Region_North America',
            'AvRating_BBB',
            'Tier_SNRFOR'
        }

        # Drop reference categories to avoid multicollinearity
        group = group.drop(columns=reference_categories, errors='ignore')

        # Prepare y
        group['LogSpread'] = np.log(group[sheet])
        y = group['LogSpread']
        
        # Prepare X
        X = group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'DocClause', 'Ccy', 'Unnamed: 0', 'Recovery', 'CompositeDepth5y'], axis=1, errors='ignore')

        # Add global factor 1
        X.insert(0, 'Constant', 1)

        # Regression
        model = sm.OLS(y, X).fit()
        
        # MSE
        y_hat = model.predict(X)
        mse = mean_squared_error(y, y_hat)
        
        # adjusted R^2
        R2 = model.rsquared_adj

        # Store the coefficients and MSE for this date
        result = {'Date': date_of_interest, 'MSE': mse, 'adjusted R2': R2}
        for coef_name, coef_value in model.params.items():
            result[coef_name] = coef_value
        
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    all_results[sheet] = results_df
    

output_file = "/Users/lufan/Documents/erasmus_classes/master blok 3/seminar/data_cleaning/nomura_regression_results.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in all_results.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

plt.figure(figsize=(12, 6))

for sheet, df in all_results.items():
    plt.plot(df['Date'], df['adjusted R2'], label=sheet)

plt.xlabel("Date")
plt.ylabel("Adjusted R²")
plt.title("Time Series of Adjusted R² for Each Maturity")
plt.legend(title="Maturity Sheets", loc="center left", bbox_to_anchor=(1, 0.5))

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the figure
output_path = "/Users/lufan/Documents/erasmus_classes/master blok 3/seminar/data_cleaning/nomura_adjusted_r2_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"Plot saved to {output_path}")

print(f"Results saved to {output_file}")
