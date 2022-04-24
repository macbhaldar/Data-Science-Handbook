# Importing the libraries
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

# Load Data
dataset=pd.read_csv('emails.csv') 
dataset.head()

# verify that there are no null columns in the dataset
dataset[dataset.isnull().any(axis=1)]

categorical = [var for var in dataset.columns if dataset[var].dtype=='O']
numerical = [var for var in dataset.columns if dataset[var].dtype!='O']
print('There are {} categorical variables : \n'.format(len(categorical)), categorical)

# check for cardinality in categorical variables
for var in categorical:
    print(var, ' contains ', len(dataset[var].unique()), ' labels')
    
# view summary statistics in numerical variables to check for outliers
print(round(dataset[numerical].describe()),2)

# Data Preprocessing
# use LabelEncoder to replace purchased (dependent variable) with 0 and 1 
from sklearn.preprocessing import LabelEncoder
dataset['Email No.']= LabelEncoder().fit_transform(dataset['Email No.'])
dataset.head()

y = dataset['Prediction']
x = dataset.drop(['Prediction'], axis=1)
print(x.head())
print(y.head())

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0) # func returns train and test data. It takes dataset and then split size test_size =0.3 means 30% data is for test and rest for training and random_state 
print(x_train.head())
print(x_test.head())
print(y_train[:10])
print(y_test[:10])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train=scaler.fit_transform(x_train) # apply on whole x data 
x_test=scaler.transform(x_test)

# Build Model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski') # by default n_neighbors = 5
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Evaluate Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, average_precision_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

cr = classification_report(y_test,y_pred)
print(cr)

accuracy_score(y_test,y_pred)

average_precision= average_precision_score(y_test,y_pred)
print(average_precision)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(classifier, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
