# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# loading the data
F_data= pd.read_csv('Fraud_check.csv')
F_data
F_data.info()

# creating a new feature for fraud if 
# taxable_income <= 30000 then putting that person in "Risky" category and others are "Good" category.
# the below code is finding good and risky category it can be done with replace function also.
x = F_data['Taxable.Income']
def risky(x):
    status=[]
    for i in x:
        if i<=30000:
            status.append('risky')
        elif i >30000:
            status.append('Good') 
    return status

fraud_status = pd.DataFrame(risky(x),columns=['status'])
fraud_status

# combining the Status column to the dataframe.
F_data['Status']= fraud_status
F_data.Status.value_counts()
sns.countplot(F_data['Status'],palette='magma');

# splitting the dataset and using label encoder to transform the object datatype to numeric datatype
le= LabelEncoder()
X= F_data.iloc[:,0:6]
Y= F_data.iloc[:,6]
X['Undergrad'] = le.fit_transform(X['Undergrad'])
X['Marital.Status'] = le.fit_transform(X['Marital.Status'])
X['Urban'] = le.fit_transform(X['Urban'])
X.drop('Taxable.Income',axis=1,inplace=True)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=45)

# Building a model on the basis of entropy
model= DecisionTreeClassifier(criterion='entropy',max_depth=8)
model.fit(X_train,Y_train)

# visualizing the tree
features=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
classes=['Good','risky']
fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,4), dpi=300)
tree.plot_tree(model,feature_names=features,class_names=classes,filled=True);

# predicting on training data
pred_on_train_data=model.predict(X_train)
# getting the count of each category 
pd.Series(pred_on_train_data).value_counts()

pd.crosstab(Y_train,pred_on_train_data)
np.mean(pred_on_train_data==Y_train)

# testing on test data
pred_on_test_data=model.predict(X_test)
pd.Series(pred_on_test_data).value_counts()

pd.crosstab(Y_test,pred_on_test_data)
np.mean(Y_test==pred_on_test_data)

# Building a model using Gini index
model1= DecisionTreeClassifier(criterion='gini',max_depth=6)
model1.fit(X_train,Y_train)
preds=model1.predict(X_test)
pd.Series(preds).value_counts()

pd.crosstab(Y_test,preds)
np.mean(Y_test==preds)
preds1=model1.predict(X_train)
pd.Series(preds1).value_counts()

pd.crosstab(preds1,Y_train)

np.mean(preds1==Y_train)

# visualizing the tree
fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize = (50,15), dpi=300)
tree.plot_tree(model1,filled=True,feature_names=X_train.columns,class_names=Y_train.unique(),fontsize=10);
