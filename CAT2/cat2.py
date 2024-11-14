import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mobileprice = pd.read_csv('MobilePriceRange.csv')

# Collums and their data types
print(mobileprice.info())

# drop any duplicates
mobileprice = mobileprice.drop_duplicates()
print(mobileprice.isnull().sum())

# EAS
label_encoder = LabelEncoder()
mobileprice['Pricerange'] = label_encoder.fit_transform(mobileprice['PriceRange'])
sns.set(style='whitegrid')

plt.figure(figsize=(8, 5))
sns.countplot(x='PriceRange', data=mobileprice, palette='Set2')
plt.title('Distribution of Mobile Price Ranges')
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Histogram for battery power
plt.figure(figsize=(10, 5))
sns.histplot(mobileprice['BatteryPower'], bins=30, kde=True, color='blue')
plt.title('Distribution of Battery Power')
plt.xlabel('Battery Power')
plt.ylabel('Frequency')

# Bar Plot for RAM vs price range
plt.figure(figsize=(12, 6))
sns.barplot(x='PriceRange', y='RAM', data=mobileprice, palette='Set2', estimator=sum)
plt.title('Average  RAM by Price Range')
plt.xlabel('Price Range')
plt.ylabel('RAM')

#Pair Plot
#sns.pairplot(mobileprice, hue='PriceRange', vars=['BatteryPower','RAM', 'InternalMemory'])
#plt.title('Pair Plot of Key features')
#plt.show()

# Question 3 on KNN models
# The fearures are batterypower and ram for price

x = mobileprice[['ClockSpeed', 'RAM']]
y = mobileprice['NoOfCores']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# create the knn classifier
knn = KNeighborsClassifier(n_neighbors=3)
# train model
knn.fit(x_train, y_train)

#make prediction
y_pred = knn.predict(x_test)

#get accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#optimise k
k_values = range(1, 4)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))

optimal_k = k_values[np.argmax(accuracies)]
print(f'Optimal k: {optimal_k}')

plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('k-NN Classifier Accuracy for Diffrent k Values')
plt.xticks(k_values)
plt.grid()
#plt.show()
