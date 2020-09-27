#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
# change the name of the column
df = df.rename(columns={"Timestamp": "Datetime"})
# convert col Datetime to_datetime
df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')

# reduction in the dimensions of the returned object
df = df.loc[:, ["Datetime", "Close"]]

print(df.tail())
