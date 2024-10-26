import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


crashList = "Airplane_Crashes_and_Fatalities_Since_1908.csv"
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


colNames = ["Date", "Location", "Aboard", "Fatalities"]

# fatality_locations = pd.DataFrame(columns=colNames)
# print(fatality_locations)

fatality_locations = pd.read_csv(crashList, usecols=colNames)
print(fatality_locations)

## No 6
high = fatality_locations.loc[fatality_locations["Fatalities"].idxmax()]
date = high["Date"]
print("Question 6.")
print(f"Date with the highest fatalities is: {date}")

## No 7
total = (fatality_locations["Fatalities"] == 0).sum()

print(f"The number of crashes with no recorded fatalities are {total}")

## No 8
fatality_locations[["Region", "U.S State.Country"]] = fatality_locations[
    "Location"
].str.split(",", n=1, expand=True)

print("Loactions have been split to Region and states")
print(f"{fatality_locations}")

## No 9

fatality_sort = fatality_locations.sort_values(by="Fatalities", ascending=False)
data = fatality_sort.head(100)

print("The first 100 records from the sorted data: ")
print(f"{data}")

## No 10

top = data.head(25)
piechart_data = top.groupby("U.S State.Country")["Fatalities"].sum()

# Drawing the pie chart
piechart_data.plot.pie(autopct="%1.2f%%")
plt.show()
