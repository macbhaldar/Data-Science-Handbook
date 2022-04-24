# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Load Data
dataset=pd.read_csv('mushrooms.csv') 
dataset.head()
dataset.describe()

# verify that there are no null columns in the dataset
dataset[dataset.isnull().any(axis=1)]
y = dataset['class']
x = dataset.drop(['class'], axis=1)
print(x.head())
print(y.head())

# Data Preprocessing
# use LabelEncoder to replace purchased (dependent variable) with 0 and 1 
from sklearn.preprocessing import LabelEncoder
y= LabelEncoder().fit_transform(y)
print(y[:10])

x = pd.get_dummies(x)
print(x.head())

Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0)
print(x_train.head())

print(x_test.head())

print(y_train[:10])

print(y_test[:10])

# Build Model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) 
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

# Evaluate Model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score
cm = confusion_matrix(y_test,y_pred)

cr = classification_report(y_test,y_pred)
print(cr)

accuracy_score(y_test,y_pred)

# Plot the decision tree
from sklearn.tree import plot_tree
plot_tree(classifier)

import graphviz # Refer to https://forum.graphviz.org/t/new-simplified-installation-procedure-on-windows/224
from sklearn import tree
dot_data = tree.export_graphviz(classifier)
graph = graphviz.Source(dot_data)
graph

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
average_precision= average_precision_score(y_test,y_pred)
print(average_precision)

disp = plot_precision_recall_curve(classifier, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

print(cm)
