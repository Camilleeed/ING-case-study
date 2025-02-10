#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:20:10 2025

@author: lufan
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import statsmodels.stats.diagnostic as smd

file_path = "clean_data2.xlsx"

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
        group = pd.get_dummies(group, columns=['Sector', 'Region', 'AvRating', 'Tier', 'Ccy', 'DocClause'], dtype=int)

        # List of reference categories
        reference_categories = {
            'Sector_Healthcare',
            'Region_North America',
            'AvRating_BBB',
            'Tier_SNRFOR'
            'Ccy_USD',
            'DocClause_MR14'
        }

        # Drop reference categories to avoid multicollinearity
        group = group.drop(columns=reference_categories, errors='ignore')

        # Prepare y
        group['LogSpread'] = np.log(group[sheet])
        y = group['LogSpread']
        
        # Prepare X
        X = group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0'], axis=1, errors='ignore')

        # Add global factor 1
        X.insert(0, 'Constant', 1)

        # Regression
        model = sm.OLS(y, X).fit()
        
        # MSE
        y_hat = model.predict(X)
        mse = mean_squared_error(y, y_hat)
        #p_values = model.pvalues
        #print('p values: ', p_values)
        #print('se: ', model.bse)
        resettest = smd.linear_reset(res=model, power=2, test_type='fitted', use_f=True)
        print(resettest)



        # Store the coefficients and MSE for this date
        result = {'Date': date_of_interest, 'MSE': mse}
        for coef_name, coef_se in model.bse.items():
            result[coef_name+'_se'] = coef_se
        for coef_name, coef_value in model.params.items():
            result[coef_name] = coef_value
        
        results_list.append(result)

    results_df = pd.DataFrame(results_list)
    all_results[sheet] = results_df

output_file = "regression_results_selected_extended.xlsx"
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in all_results.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Results saved to {output_file}")













