import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

"""
    This is the code for lab 6
        1. first we clean the data
        2. Then we do the analysis
"""
# Load data from the file to a data frame
df = pd.read_csv("Mall_Customers.csv")

# Remove Duplicates
df = df.drop_duplicates()

# Get the basic stats
print(df.describe())

# now we plot the data
# Histogram
plt.figure(figsize=(15, 6))
df["Age"] = pd.cut(
    df["Age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=["18-25", "26-35", "36-50", "51-65", "65+"],
)

sns.histplot(data=df, x="Spending Score (1-100)", hue="Age", multiple="dodge")
# a pie chart of gender in data
gender = df["Gender"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    gender,
    labels=gender.index,
    autopct="%1.1f%%",
    colors=["pink", "lightblue"],
    explode=(0.05, 0.05),
    shadow=True,
)
plt.title("Gender Distribution Of Mall Customers")
plt.axis("equal")


dfk = pd.read_csv("Mall_Customers.csv")
x = dfk[["Spending Score (1-100)", "Age"]].values
u = dfk[["Age"]].values

# scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
u_scaled = scaler.fit_transform(u)


# elbow method for one feature
inertias = []
i_range = range(1, 10)

for i in i_range:
    imeans = KMeans(n_clusters=i)
    imeans.fit(u_scaled)
    inertias.append(imeans.inertia_)

# plot
plt.figure(figsize=(15, 6))
plt.plot(i_range, inertias, "bx-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Curve for Age")
plt.grid(True)

# elbow Method
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    inertias.append(kmeans.inertia_)

# plot elbow curve
plt.figure(figsize=(15, 6))
plt.plot(k_range, inertias, "bx-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (within Cluster Sum Of Squares)")
plt.title("Elbow Method For Age vs Spending")
plt.grid(True)

# silhouette Analysis
max_clusters = 10
min_clusters = 1
sil_score = []
k_range_sil = range(2, 11)
# claculating the silhouette scores

for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    score = silhouette_score(x_scaled, kmeans.labels_)
    sil_score.append(score)

# Plot silhouette scores
plt.figure(figsize=(15, 6))
plt.plot(k_range_sil, sil_score, "ro-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis: Finding Optimal k")
plt.xticks(k_range_sil)
plt.grid(True)


# 1 Feature
sil_score_1 = []
i_range_sil = range(2, 11)


for i in i_range_sil:
    kmeans = KMeans(n_cluster=i)
    kmeans.fit(u_scaled)
    score = silhouette_score(u_scaled, kmeans.labels_)
    sil_score_1.append(score)

# plot
plt.figure(figsize=(15, 6))
plt.plot(i_range_sil, sil_score_1, "ro-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis: Finding Optimal k")
plt.xticks(i_range_sil)
plt.grid(True)

# show
plt.show()
