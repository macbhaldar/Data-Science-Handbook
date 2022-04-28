# importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# importing datasets
salary_data = pd.read_csv('SalaryData_Train.csv')
salary_test_data = pd.read_csv('SalaryData_Test.csv')
salary_data.head()

# mapping the Target
salary_data['Salary'] = salary_data['Salary'] .map({' <=50K' : 0, ' >50K' : 1})
salary_test_data['Salary'] = salary_test_data['Salary'].map({' <=50K' : 0, ' >50K' : 1})
salary_test_data

salary_data['Salary'].value_counts()

# Transforming the text data into numerical data 
le = LabelEncoder()
text_colums = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
for i in text_colums:
    salary_data[i] = le.fit_transform(salary_data[i])
    salary_test_data[i] = le.fit_transform(salary_test_data[i])

# normalized data
salary_test_data
salary_data.describe()

# EDA
sns.set(rc={'figure.figsize':(8,5)})
c1= salary_data.columns
c = c1.drop("age")
c

for i in salary_data.columns:
    plt.title("Distribution of {}".format(i))
    sns.displot(salary_data[i],color='#CD6090')
    
# correlation
corr = salary_data.corr()
sns.heatmap(corr,cmap='pink_r');

X_train = salary_data.drop('Salary',axis=1)
Y_train = salary_data['Salary']
X_test = salary_test_data.drop('Salary',axis=1)
Y_test = salary_test_data['Salary']

print("Shape of X_train is" + str(X_train.shape))
print("Shape of X_test is " + str(X_test.shape))

plt.figure(figsize=(16,5))
print("Skew: {}".format(salary_data['age'].skew()))
print("Kurtosis: {}".format(salary_data['age'].kurtosis()))
ax = sns.kdeplot(salary_data['age'],shade=True,color='g')
plt.show()

plt.figure(figsize=(16,5))
print("Skew: {}".format(salary_data['capitalgain'].skew()))
print("Kurtosis: {}".format(salary_data['capitalgain'].kurtosis()))
ax = sns.kdeplot(salary_data['capitalgain'],shade=True,color='red')
plt.show()

# Model Building
classifier = GaussianNB()
classifier.fit(X_train,Y_train)
## GaussianNB()

# testing on training data 
a1=accuracy_score(Y_train,classifier.predict(X_train))*100
print("The Accuracy on Training data is : {}".format(round(a1)))
## The Accuracy on Training data is : 80

# testing on testing data 
a2=accuracy_score(Y_test,classifier.predict(X_test))*100
print("The Accuracy on Test data is : {}".format(round(a2)))
## The Accuracy on Test data is : 79
