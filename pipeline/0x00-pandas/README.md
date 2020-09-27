#0x00. Pandas
Specializations - Machine Learning - The Pipeline

###General

    What is pandas?
    What is a pd.DataFrame? How do you create one?
    What is a pd.Series? How do you create one?
    How to load data from a file
    How to perform indexing on a pd.DataFrame
    How to use hierarchical indexing with a pd.DataFrame
    How to slice a pd.DataFrame
    How to reassign columns
    How to sort a pd.DataFrame
    How to use boolean logic with a pd.DataFrame
    How to merge/concatenate/join pd.DataFrames
    How to get statistical information from a pd.DataFrame
    How to visualize a pd.DataFrame

Requirements

Download Pandas 0.24.x
```
pip install --user pandas
```

###Tasks

**0. From Numpy mandatory**

Write a function def from_numpy(array): that creates a pd.DataFrame from a np.ndarray.

---
**1. From Dictionary**

Write a python script that created a pd.DataFrame from a dictionary.

---
**2. From File**

Write a function def from_file(filename, delimiter): that loads data from a file as a pd.DataFrame.

---
**3. Rename**

Complete the script to perform the following:

    Rename the column Timestamp to Datetime
    Convert the timestamp values to datatime values
    Display only the Datetime and Close columns

---
**4. To Numpy**

Complete the script to take the last 10 rows of the columns High and Close and
convert them into a numpy.ndarray.

---
**5. Slice**

Complete the script to slice the pd.DataFrame along the columns High, Low,
Close, and Volume_BTC, taking every 60th row.

---
**6. Flip it and Switch it**

Complete the script to alter the pd.DataFrame such that the rows and columns
are transposed and the data is sorted in reverse chronological order.

---
**7. Sort**

Complete the script to sort the pd.DataFrame by the High price in descending
order.

---
**8. Prune mandatory**

Complete the script to remove the entries in the pd.DataFrame where
Close is NaN.

---
**9. Fill**
Complete the script to fill in the missing data points in the
pd.DataFrame.

___
**10. Indexing**

Complete the script to index the pd.DataFrame on the Timestamp column.

---
**11. Concat**

Complete the script to index the pd.DataFrames on the Timestamp
columns and concatenate them.

---
**12. Hierarchy**

Based on 11-concat.py, rearrange the MultiIndex levels such that
timestamp is the first level.

---
**13. Analyze**

Complete the script to calculate descriptive statistics for all
columns in pd.DataFrame except Timestamp.

---
**14. Visualize**

Complete the script to visualize the pd.DataFrame.




