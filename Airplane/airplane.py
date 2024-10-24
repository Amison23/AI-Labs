import pandas as pd
import numpy as np

crashList = "Airplane/Airplane_Crashes_and_Fatalities_Since_1908.csv"
list = pd.read_csv(crashList)

# Number of rows and columns
rows = len(list)
columns = len(list.columns)

# print(f"columns: {columns}, rows: {rows}")

last = list.tail(75)
print(f"Last 75 rows: {last}")

columnNames = list.columns
print(f"{columnNames}")

# Treating missing data depends on the column type data
# - Numerical Data: mean or median imputaiontion 
# is used because mean or median can be used to replace 
# missing data so this would be used in Fatalities for instance

# - Numerical Data, High miss rate: in such scenarios, dropping the column 
# would work, e.g. cn/in or regressing imputation if there's strong correlation with 
# other columns e.g Time, Registration, Route, Flight No'. These columns can be 
# cross checked with flight data logs at airports or airline distributors and filled out



# - Categorical Data: this works well with categorical data like 
# Registration, Type and Route becuase the most common value within 
# the data can be used to replace the missing data

# - Categorical with High missing data: For this a new category can be made.
# This can work well with the Summary, Route as well and the Flight No'.


colNames = ["Date", "Location", "Abroad", "Fatalities"]

fatality_locations = pd.DataFrame(columns=colNames)
print(fatality_locations)
