# Importing the required libraries
import numpy as np
import os
from data import getData
from sklearn import svm
from sklearn.metrics import *
from warnings import simplefilter
from sklearn.decomposition import PCA

# ignore all warnings
simplefilter(action='ignore')

X, y = getData()


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 10))
sns.heatmap(np.corrcoef(X, rowvar=False), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Sensor Readings')
plt.show()
