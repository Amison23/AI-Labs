import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
    This is the code for lab 6
        1. first we clean the data
        2. Then we do the analysis
"""
## Load data from the file to a data frame
df = pd.read_csv("Mall_Customers.csv")

# Remove Duplicates
df = df.drop_duplicates()

# Get the basic stats
print(df.describe())

# now we plot a boxplot of the data
# sns.histplot(df["Age"])
# sns.histplot(df["Spending Score (1-100)"])
plt.figure(figsize=(15, 6))
sns.boxplot(
        data=df,
        x='Age',
        y='Spending Score (1-100)'
        )
plt.title('Spending Distribution by Age Group')
plt.xticks(rotation=45)

# Histogram
plt.figure(figsize=(15,6))
df['Age'] = pd.cut(df['Age'],
                   bins=[0, 25, 35, 50, 65,100],
                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])

sns.histplot(
        data=df,
        x='Spending Score (1-100)',
        hue='Age',
        multiple='dodge'
        )


# plot
plt.show()
