from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from Dataloader import DataLoader
from sklearn import preprocessing
from sklearn.svm import SVC
import pandas as pd
import numpy as np


def countAnamoly(y_pred):
    count = 0
    for i in y_pred:
        if i == True:
            count += 1
    # Number of predicted anamoly
    print(count)

DataLoader = DataLoader()
data = DataLoader.get_dataset()

# Using SVM to create models with several trails
# Normalize the data
X = data.drop('result', axis=1)
y = data['result']
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(x_scaled)

inputs = [0.01, 0.1, 1, 10, 100]
for i in inputs:
    svclassifier = SVC(kernel = 'rbf', C = i, class_weight="balanced")
    svclassifier.fit(X, y)
    y_pred = svclassifier.predict(X)
    print("SVM with penalty: ", i)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # print out the number of predicted anomoly
    # countAnamoly(y_pred)

    # Use logistic regression
    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression(C = i)
    logreg.fit(X, y)
    counts2 = logreg.predict(X)
    print("logistic regression with penalty: ", i)
    print(confusion_matrix(y, counts2))
    print(classification_report(y, counts2))
    # countAnamoly(counts2)


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=20, 
                               bootstrap = True,
                               max_features = 'sqrt')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit on training data
model.fit(X_train, y_train)
pred_y = model.predict(X_test)
print(confusion_matrix(y_test, pred_y))
print(classification_report(y_test, pred_y))
print(y_test[y_test == True].count())
# print()

