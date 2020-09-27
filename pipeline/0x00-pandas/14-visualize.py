#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Location of data based on Unix Timestamp in seconds
df = df.loc[df['Timestamp'] > 1483185599]
df = df.drop(['Weighted_Price'], axis=1)

# change the name of the column
df = df.rename(columns={"Timestamp": "Date"})
# convert col Datetime to_datetime
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df = df.set_index("Date")

# missing values set to previous row's Close value
df["Close"].fillna(method="ffill", inplace=True)
df["High"].fillna(value=df.Close.shift(1), inplace=True)
df["Low"].fillna(value=df.Close.shift(1), inplace=True)
df["Open"].fillna(value=df.Close.shift(1), inplace=True)
df["Volume_(BTC)"].fillna(value=0, inplace=True)
df["Volume_(Currency)"].fillna(value=0, inplace=True)

# plot at daily intervals: resample data to days
df_plot = pd.DataFrame()
df_plot = (df.resample('D')
           .agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                 'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum',
                 'Close': 'last'}))

df_plot.plot()
plt.show()
