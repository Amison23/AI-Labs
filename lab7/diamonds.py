import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

plt.style.use("ggplot")


data = "diamonds.csv"
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
    df["carat"], bins=[0, 0.4, 0.7, 1], labels=["0.0-0.4", "0.5-0.7", "0.8-1"]
)
carat = df["carat"].value_counts()
plt.pie(carat, colors=["blue", "green", "red"], labels=carat.index)
plt.title("Pie Chart for carat range from 0.0 - 1")


# 3rd figure
plt.figure(figsize=(10, 6))
grouped = df.groupby("carat")["price"].mean()
grouped.plot(kind="bar")
plt.title("Average Price for Each Carat Category")
plt.xlabel("Carat Category")
plt.ylabel("Average Price")


# plt.show()

# Modelling dataset
diamonds_model = pd.read_csv("diamonds.csv")
diamonds_model = diamonds_model.sample(n=5000)

# linear regression
le = LabelEncoder()
collumns = ["cut", "color", "clarity"]
for col in collumns:
    diamonds_model[col] = le.fit_transform(diamonds_model[col])

x = diamonds_model[["carat", "cut", "color", "clarity"]]
y = diamonds_model["price"]

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(x_train, y_train)

# make Prediction
y_pred = model.predict(x_test)

# Model performance
print("\nModel Performance:")
print(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")


# visualize

plt.figure(figsize=(15, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "b-", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Prices")
plt.tight_layout()

# plt.show()


# PCA modelling
collumns2 = ["carat", "depth", "table", "x", "y", "z"]
x = diamonds_model[collumns2]
y = diamonds_model["price"]

# Standardize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

# show ratio variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_)}")

# show components loadings
components_df = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=collumns2)
print(f"\nPCA Components: \n {components_df}")

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, test_size=0.1, random_state=42
)

# train model
model2 = LinearRegression()
model2.fit(x_train, y_train)

# make predictions
y_pred = model2.predict(x_test)

# calculate metrics
print(f"\nModel Performance: \nR^2 Score: {r2_score(y_test, y_pred): .3f} ")

# visuallise
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap="viridis", alpha=0.5)
plt.colorbar(label="Price")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA Component vs Price")

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b-', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')

plt.tight_layout()

plt.show()
