import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from datetime import datetime, timedelta


import numpy as np
import pandas as pd


out_of_sample_data = pd.read_excel('markit_out_of_sample_202302_202303_no_spread.xlsx')
in_sample_data = pd.read_excel('markit_in_sample_202302_202303.xlsx')

tenor = ['6m', '1y', '2y', '3y', '4y', '5y', '7y', '10y', '15y', '20y', '30y']

# Set dates into date format
in_sample_data['Date'] = pd.to_datetime(in_sample_data['Date'])
out_of_sample_data['Date'] = pd.to_datetime(in_sample_data['Date'])

Dates = out_of_sample_data['Date'].drop_duplicates()
in_sample_data = in_sample_data.set_index('Date', drop=False)

writer = pd.ExcelWriter('out_of_sample_data_AvRating.xlsx')
i = 0
j = 0

for index, row in out_of_sample_data.iterrows(): # iterate every row
    date = row['Date']
    ticker_name = row['Ticker']
    df = in_sample_data[(in_sample_data['Ticker'] == ticker_name)] # select the same company
    df_today = df[df['Date'] == date][['Date', 'AvRating']] # select the same date
    df_yesterday = df[df['Date'] == date - timedelta(days=1)][['Date', 'AvRating']] # the day before
    df_tomorrow = df[df['Date'] == date + timedelta(days=1)][['Date', 'AvRating']] # the day after
    if not df_today.empty:
        i = i + 1
        out_of_sample_data.loc[index, 'AvRating'] == df_today['AvRating'].iloc[-1]
    elif not df_yesterday.empty and not df_tomorrow.empty and df_yesterday['AvRating'].iloc[-1] == df_tomorrow['AvRating'].iloc[-1]:
        out_of_sample_data.loc[index, 'AvRating'] == df_yesterday['AvRating'].iloc[-1]
        j = j + 1

print(i)
print(j)

out_of_sample_data.to_excel(writer)
writer.close()


















































