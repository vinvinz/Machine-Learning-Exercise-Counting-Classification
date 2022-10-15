from re import X
import pandas as pd
import mediapipe
import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('test_dataset.csv')

X = df.values
y = df['label'].values

# X_train, X_temp, Y_train, Y_temp = train_test_split(X, y,train_size=.8 random_state = 1)
# X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, train_size=.5, shuffle=False)

model = KNeighborsClassifier()
model.fit(X, y)


y_pred = model.predict(X)
print((y==y_pred).sum())
print((y==y_pred).sum()/y.shape[0])
print(model.score(X, y))