import numpy as np
import os
from data import getData
from sklearn import svm
from sklearn.metrics import *
from warnings import simplefilter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 


X, y = getData()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=1)




knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test, y_pred,average='weighted'))
print(precision_score(y_test, y_pred, average='weighted'))
print(recall_score(y_test, y_pred, average='macro'))


# from sklearn.metrics import confusion_matrix
# print (confusion_matrix(y_test, y_pred))