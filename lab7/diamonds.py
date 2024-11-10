import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

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

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "b-", lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")

plt.tight_layout()
X = diamonds_model[["carat", "depth", "table", "x", "y", "z"]]
y = diamonds_model["price"]
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize models
models = {"Linear Regression": LinearRegression(), "Lasso": Lasso(), "Ridge": Ridge()}

# Define parameter grids for Lasso and Ridge
param_grid_lasso = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_ridge = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

# Perform GridSearchCV for Lasso and Ridge
lasso_grid = GridSearchCV(
    Lasso(), param_grid_lasso, cv=5, scoring="neg_mean_squared_error"
)
ridge_grid = GridSearchCV(
    Ridge(), param_grid_ridge, cv=5, scoring="neg_mean_squared_error"
)
# Train and evaluate models
results = {}
predictions = {}
for name, model in models.items():
    if name == "Lasso":
        model = lasso_grid
    elif name == "Ridge":
        model = ridge_grid

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
predictions[name] = y_pred

# Calculate metrics
r2 = r2_score(y_test, y_pred)

# Perform cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
results[name] = {
    "R2": r2,
    "CV_mean": cv_scores.mean(),
    "CV_std": cv_scores.std(),
}

# Print best parameters for Lasso and Ridge
if name in ["Lasso", "Ridge"]:
    print(f"\n{name} Best Parameters:")
    print(f"Alpha: {model.best_params_['alpha']}")

# Print results
print("\nModel Comparison:")
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"R² Score: {metrics['R2']:.3f}")
    print(
        f"Cross-validation R² Score: {metrics['CV_mean']:.3f} (+/- {metrics['CV_std']*2:.3f})"
    )

# Visualizations
plt.figure(figsize=(15, 5))

# Plot 1: Actual vs Predicted for all models
plt.subplot(1, 2, 1)
for name, y_pred in predictions.items():
    plt.scatter(y_test, y_pred, alpha=0.5, label=name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()


# Feature importance plot for Lasso and Ridge
plt.figure(figsize=(12, 5))

# Plot Ridge coefficients
plt.subplot(1, 2, 2)
ridge_coef = ridge_grid.best_estimator_.coef_
plt.bar(X.columns, ridge_coef)
plt.title("Ridge Coefficients")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
