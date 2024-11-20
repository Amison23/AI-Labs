import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

mobiledata = pd.read_csv('MobilePriceRange.csv')

print(mobiledata.head())
print(mobiledata.isnull().sum())
print(mobiledata.describe())

# Label Encoding for 'price_range'
label_encoder = LabelEncoder()
mobiledata['PriceRange'] = label_encoder.fit_transform(mobiledata['PriceRange'])

# Visualizations
# Distribution of the target variable (price range)
sns.countplot(x='BatteryPower', data=mobiledata)
plt.title('Distribution of Mobile Batterry')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(mobiledata.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Feature selection (let's say we select 'feature1' and 'feature2')
X = mobiledata[['RAM', 'ClockSpeed']]
y = mobiledata['NoOfCores']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Finding the optimal k using cross-validation
k_values = range(1, 21)
accuracy = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy.append(knn.score(X_test, y_test))

optimal_k = k_values[accuracy.index(max(accuracy))]
print(f'Optimal k: {optimal_k}')

# Create and evaluate the k-NN model
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Random Forest Classifier
rf = RandomForestClassifier()
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
rf_grid = GridSearchCV(rf, rf_params, cv=5,n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f'Best parameters for Random Forest: {rf_grid.best_params_}')

# Evaluate Random Forest
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

svm = SVC()
svm_params = {
    'C': [0.1, 1.0],
    'kernel': ['linear']
}
svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
print(f'Best parameters for SVM: {svm_grid.best_params_}')

# Evaluate SVM
svm_best = svm_grid.best_estimator_
y_pred_svm = svm_best.predict(x_test_scaled)
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Comparison of models
models = ['k-NN', 'Random Forest', 'SVM']
predictions = [y_pred, y_pred_rf, y_pred_svm]

for i, model in enumerate(models):
    print(f'Confusion Matrix for {model}:')
    print(confusion_matrix(y_test, predictions[i]))
    print(f'Classification Report for {model}:')
    print(classification_report(y_test, predictions[i]))

