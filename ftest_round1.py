import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.api import anova_lm

# File paths
file_path = "Nomura_extended_data.xlsx"
output_files = {
    "Ccy": "./variable_f_test/full_ex_ccy_f.xlsx",
    "DocClause": "./variable_f_test/full_ex_docclause_f.xlsx",
    "Recovery": "./variable_f_test/full_ex_recovery_f.xlsx",
    "CompositeDepth5y": "./variable_f_test/full_ex_composite_f.xlsx"
}
output_plot_paths = {
    "Ccy": "./variable_f_test/ftest_ccy_plot.png",
    "DocClause": "./variable_f_test/ftest_docclause_plot.png",
    "Recovery": "./variable_f_test/ftest_recovery_plot.png",
    "CompositeDepth5y": "./variable_f_test/ftest_composite_plot.png"
}

# Load Excel file
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Dictionary to store results
all_results = {key: {} for key in output_files.keys()}

# Loop through each sheet
for sheet in sheet_names:
    data = pd.read_excel(file_path, sheet_name=sheet)
    data['Date'] = pd.to_datetime(data['Date'])
    
    results_list = {key: [] for key in output_files.keys()}

    for date_of_interest in data['Date'].unique():
        df = data[data['Date'] == date_of_interest]
        
        # Create dummy variables
        # group = pd.get_dummies(group, columns=['Sector', 'Region', 'AvRating', 'Tier','DocClause','Ccy'], dtype=int)
        # reference_categories = {'Sector_Healthcare', 'Region_North America', 'AvRating_BBB', 'Tier_SNRFOR', 'DocClause_MR14','Ccy_JPY'}
        # group = group.drop(columns=reference_categories, errors='ignore')
        
        # Prepare dependent variable (LogSpread)
        df['LogSpread'] = np.log(df[sheet])
        y = df['LogSpread']

        models = {}
        columns_selected = ['Sector', 'Region', 'AvRating', 'Tier','DocClause','Ccy']
        group = pd.get_dummies(df, columns=['Sector', 'Region', 'AvRating', 'Tier','DocClause','Ccy'], dtype=int)
        reference_categories = {'Sector_Healthcare', 'Region_North America', 'AvRating_BBB', 'Tier_SNRFOR', 'DocClause_MR14','Ccy_JPY'}
        group = group.drop(columns=reference_categories, errors='ignore')
        models['full'] = group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0'], axis=1, errors='ignore')

        for key in output_files.keys():
            columns_selected = ['Sector', 'Region', 'AvRating', 'Tier','DocClause','Ccy']
            if key in columns_selected:
                columns_selected.remove(key)
            group = pd.get_dummies(df, columns=columns_selected, dtype=int)
            reference_categories = {'Sector_Healthcare', 'Region_North America', 'AvRating_BBB', 'Tier_SNRFOR', 'DocClause_MR14','Ccy_JPY'}
            group = group.drop(columns=reference_categories, errors='ignore')
            models[key] = group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0', key], axis=1, errors='ignore')


        # Define models
        #models = {
        #    "full": group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0'], axis=1, errors='ignore'),
        #    "ccy": group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0','Ccy'], axis=1, errors='ignore'),
        #    "docclause": group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0','DocClause'], axis=1, errors='ignore'),
        #    "recovery": group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0','Recovery'], axis=1, errors='ignore'),
        #    "composite": group.drop(['Date', sheet, 'LogSpread', 'Ticker', 'ShortName', 'Country', 'Unnamed: 0','Composite5y'], axis=1, errors='ignore')
        #}
        
        for key in output_files.keys():
            X_full = sm.add_constant(models["full"])
            X_reduced = sm.add_constant(models[key])
            print(key)
            print(models[key])
            model_full = sm.OLS(y, X_full).fit()
            model_reduced = sm.OLS(y, X_reduced).fit()
            
            # Perform F-test
            anova_results = anova_lm(model_reduced, model_full)
            f_statistic = anova_results['F'][1]
            p_value = anova_results['Pr(>F)'][1]
            print(p_value)
            
            # Store results
            results_list[key].append({
                'Date': date_of_interest,
                'F_statistic': f_statistic,
                'p_value': p_value
            })
    
    # Convert results to DataFrame
    for key in output_files.keys():
        all_results[key][sheet] = pd.DataFrame(results_list[key])

# Save results to Excel
for key, file_path in output_files.items():
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in all_results[key].items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Plot p-values over time
for key, plot_path in output_plot_paths.items():
    plt.figure(figsize=(12, 6))
    for sheet, df in all_results[key].items():
        plt.plot(df['Date'], df['p_value'], label=sheet)
    
    plt.xlabel("Date")
    plt.ylabel("p-value")
    plt.title(f"Time Series of F-test p-value for Full vs. Exclude {key.capitalize()}")
    plt.legend(title="Maturity Sheets", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Results saved to {output_files[key]}")
    print(f"Plot saved to {plot_path}")
