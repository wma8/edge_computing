from Dataloader import DataLoader
from sklearn import preprocessing
import pandas as pd
import numpy as np
# Split data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DataLoader = DataLoader()
data = DataLoader.get_dataset()

# Using SVM to create models with several trails
# Normalize the data
X = data.drop('result', axis=1)
y = data['result']
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)

svclassifier = SVC(kernel = 'rbf', class_weight="balanced")
svclassifier.fit(X, y)
y_pred = svclassifier.predict(X)
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

count = 0
for i in y_pred:
    if i == True:
        count += 1
# Number of predicted anamoly
print(count)

