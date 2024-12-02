# Importing the required libraries
import numpy as np
import os
from data import getData
from sklearn import svm
from sklearn.metrics import *
from warnings import simplefilter

# ignore all warnings
simplefilter(action='ignore')

X, y = getData()


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(np.shape(X))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LogisticRegression


# from sklearn.linear_model import LogisticRegression  
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print('LogisticRegression')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # SVM
# from sklearn.svm import SVC

# clf = svm.SVC()

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)


# print("SVC")
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LGBM
# from lightgbm import LGBMClassifier
 
# lgbm = LGBMClassifier(verbose=-1)

# lgbm.fit(X_train, y_train)

# y_pred = lgbm.predict(X_test)



# print('LGBMClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Gaussian

# from sklearn.naive_bayes import GaussianNB

# gnb = GaussianNB()

# gnb.fit(X_train, y_train)

# y_pred = gnb.predict(X_test)

# print('GaussianNB')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Multinomial


# from sklearn.naive_bayes import MultinomialNB

# # Initialize the Gaussian Naive Bayes classifier
# nb = MultinomialNB()

# nb.fit(X_train, y_train)

# y_pred = nb.predict(X_test)

# print('MultinomialNB')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Stochastic Gradient Descent Classifier

# from sklearn import linear_model

# SGDClf = linear_model.SGDClassifier()
# SGDClf.fit(X_train, y_train)

# y_pred = SGDClf.predict(X_test)


# print('SGDClf')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # KNeighbors 


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(metric='cosine', n_neighbors= 11)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


print('KNeighborsClassifier')
print(accuracy_score(y_test,y_pred))
print(f1_score(y_test, y_pred,average='weighted'))
print(precision_score(y_test, y_pred, average='weighted'))
print(recall_score(y_test, y_pred, average='macro'))
# 0.9810250391236307
# 0.981012850524513
# 0.9811271388793069
# 0.9812841845489705
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, y_pred))
# # KNeighborsClassifier Accuracy: 98.865%
# [[1264    3   26    0]
#  [   2 1279   11    0]
#  [  11   44 1243    0]
#  [   0    0    0 1229]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Decision Trees 
# from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)

# y_pred = dt.predict(X_test)

# print('DecisionTreeClassifier')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Random Forest 

# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)


# print('RandomForestClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier()
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print ('GradientBoostingClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))
# print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   LinearDiscriminantAnalysis

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print('LinearDiscriminantAnalysis')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   MLPClassifier

# from sklearn.neural_network import MLPClassifier
# mlp_classifier = MLPClassifier()
# mlp_classifier.fit(X_train,y_train)
# y_pred = mlp_classifier.predict(X_test)

# print('LinearDiscriminantAnalysis')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))