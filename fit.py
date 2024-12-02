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
# classifier = LogisticRegression(C= 100.0, solver= 'sag')
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print('LogisticRegression')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))
# 0.939358372456964
# 0.9399032943814696
# 0.9411863135672173
# 0.9401391114773148

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # SVM
# from sklearn.svm import SVC

# clf = svm.SVC(C= 100, kernel= 'poly')

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)


# print("SVC")
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.9559859154929577
# 0.9561667881122269
# 0.9567303240475821
# 0.9565644076772889
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LGBM
# from lightgbm import LGBMClassifier
 
# lgbm = LGBMClassifier(learning_rate= 0.05, n_estimators= 100)

# lgbm.fit(X_train, y_train)

# y_pred = lgbm.predict(X_test)



# print('LGBMClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))


# 0.9921752738654147
# 0.9921701418523057
# 0.9921869709643948
# 0.9922833209703807
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Gaussian

# from sklearn.naive_bayes import GaussianNB

# gnb = GaussianNB(var_smoothing= 1e-09)

# gnb.fit(X_train, y_train)

# y_pred = gnb.predict(X_test)

# print('GaussianNB')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.7255477308294209
# 0.7404523953992131
# 0.7741631142318668
# 0.7266054104955815
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Naive Bayes: Multinomial


# from sklearn.naive_bayes import MultinomialNB

# # Initialize the Gaussian Naive Bayes classifier
# nb = MultinomialNB(alpha= 0.5)

# nb.fit(X_train, y_train)

# y_pred = nb.predict(X_test)

# print('MultinomialNB')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.6625586854460094
# 0.6608203194446226
# 0.6636984523019439
# 0.6668204825293655
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Stochastic Gradient Descent Classifier

# from sklearn import linear_model

# SGDClf = linear_model.SGDClassifier(alpha= 0.0001, loss= 'hinge')
# SGDClf.fit(X_train, y_train)

# y_pred = SGDClf.predict(X_test)


# print('SGDClf')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.8587636932707355
# 0.8568874176161684
# 0.8569510487196652
# 0.8606685003050994
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # KNeighbors 


# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)


# print('KNeighborsClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))
# from sklearn.metrics import confusion_matrix
# print (confusion_matrix(y_test, y_pred))
# # KNeighborsClassifier Accuracy: 98.865%
# # [[1280    4    9    0]
# #  [   0 1286    6    0]
# #  [   4   35 1259    0]
# #  [   0    0    0 1229]]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Decision Trees 
# from sklearn.tree import DecisionTreeClassifier

# dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 15)
# dt.fit(X_train, y_train)

# y_pred = dt.predict(X_test)

# print('DecisionTreeClassifier')

# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.9831768388106417
# 0.9831545463667173
# 0.983356896172061
# 0.9834144460364027
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # Random Forest 

# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(max_features= 'log2', n_estimators= 300)
# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)


# print('RandomForestClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.9943270735524257
# 0.9943231931468207
# 0.9943347580830304
# 0.9944062876166476

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingClassifier

# clf = GradientBoostingClassifier(learning_rate= 0.2, max_depth= 3, min_samples_leaf=1, min_samples_split= 2, n_estimators= 300, subsample= 1.0)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print ('GradientBoostingClassifier')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.9939358372456965
# 0.9939357521841964
# 0.9939441803044067
# 0.9940182466159989

# print("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   LinearDiscriminantAnalysis

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis(n_components= None, shrinkage= None, solver= 'svd', tol= 0.001)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)

# print('LinearDiscriminantAnalysis')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))
# 0.8073161189358372
# 0.8128595456772352
# 0.8377884364295032
# 0.8075404941656986
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   MLPClassifier

# from sklearn.neural_network import MLPClassifier
# mlp_classifier = MLPClassifier(activation= 'logistic', alpha = 0.01, hidden_layer_sizes = (100,), learning_rate = 'invscaling', solver= 'lbfgs')
# mlp_classifier.fit(X_train,y_train)
# y_pred = mlp_classifier.predict(X_test)

# print('LinearDiscriminantAnalysis')
# print(accuracy_score(y_test,y_pred))
# print(f1_score(y_test, y_pred,average='weighted'))
# print(precision_score(y_test, y_pred, average='weighted'))
# print(recall_score(y_test, y_pred, average='macro'))

# 0.965962441314554
# 0.9660815976894258
# 0.9665745021097717
# 0.9663972005467332