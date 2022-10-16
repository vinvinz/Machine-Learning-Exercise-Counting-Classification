import pandas as pd
import mediapipe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

df = pd.read_csv('test_dataset.csv')

X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

#Train Machine Learning Classification Model
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'sv':make_pipeline(StandardScaler(), svm.SVC()),
    'kn':make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

#Evaluate and Serialized Model
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, "Accuracy:", accuracy_score(y_test.values, yhat),
          "Precision:", precision_score(y_test.values, yhat, average='macro'),
          "Recall:", recall_score(y_test.values, yhat, average='macro'))

# # model = svm.SVC()
# # model = KNeighborsClassifier(n_neighbors=3)
# model = LogisticRegression()
# # model.fit(X, y)

# # y_pred = model.predict(X_test[0])
# # print((y==y_pred).sum())
# # print((y==y_pred).sum()/y.shape[0])
# # print(model.score(X, y))


#Print Confusion Matrix of the trained model

# model = fit_models['sv']
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred))
# df_cm = confusion_matrix(y_test, y_pred)
# print(model.score(X_train, y_train))

# # Plot the confution matrix of 4 classes in the Model
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 32}) # font size

# plt.show()


#Save Trained model in pickle

# model = fit_models['kn']
# model.fit(X_train, y_train)

# saved_model = pickle.dumps(model)

# with open('exercise.pkl', 'wb') as f:
#     pickle.dump(model, f)


#Load trained model with the pkl file

# with open('exercise.pkl', 'rb') as f:
#     model = pickle.load(f)