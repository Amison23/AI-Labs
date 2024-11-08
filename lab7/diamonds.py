import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')



data = "lab7/diamonds.csv"
df = pd.read_csv(data)

# data understanding
# print(df.shape)
print(df.describe())

# cleaning....
# print(df.head)
df = df.drop_duplicates().dropna()

# print(df.dtypes)

# Features....
plt.figure(figsize=(8, 3))
df["carat"] = pd.cut(
  df["carat"],
  bins=[0, 0.4, 0.7, 1],
  labels=['0.0-0.4','0.5-0.7', '0.8-1'])


carat = df['carat'].value_counts()
plt.figure()
plt.pie(
  carat,
  colors=["blue", "green", "red"]
)






