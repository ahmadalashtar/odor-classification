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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load dataset

# Initialize k-NN classifier
knn = KNeighborsClassifier(n_neighbors=1)

# Set up k-fold cross-validation
k = 3
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

# Get cross-validated predictions
y_pred = cross_val_predict(knn, X, y, cv=kf)
scores = cross_val_score(knn, X, y, cv=kf)
# Calculate metrics
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores)}")
# Print the results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
