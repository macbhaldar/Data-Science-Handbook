# loading the dependecies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# loading the data
fraud_data= pd.read_csv('Fraud_check.csv')
fraud_data

# Data preprocessing
# algorithm to classify good and risky customers
def risky(x):
    status=[]
    for i in x:
        if i<=30000:
            status.append('risky')
        elif i >30000:
            status.append('Good') 
    return status
a=fraud_data['Taxable.Income']
fraud_status = pd.DataFrame(risky(a),columns=['status'])
fraud_data['status']=fraud_status
fraud_data

fraud_data.rename(columns={'Marital.Status': 'Marital_Status'},inplace=True)

# label enocoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
fraud_data['Undergrad']= le.fit_transform(fraud_data['Undergrad'])
fraud_data['Marital_Status']= le.fit_transform(fraud_data['Marital_Status'])
fraud_data['Urban']= le.fit_transform(fraud_data['Urban'])
fraud_data['status']= le.fit_transform(fraud_data['status'])
fraud_data

# Data Visulization
plt.title("Maritial Status of the data points")
sns.countplot(fraud_data['Marital_Status']);

sns.distplot(fraud_data['Taxable.Income'],color='orange');

sns.histplot(fraud_data['Work.Experience'],color='orange');

sns.scatterplot(y='Taxable.Income',x='Work.Experience',data=fraud_data,hue='status')

sns.distplot(fraud_data['City.Population'],color='orange');

X=fraud_data.iloc[:,0:6]
Y=fraud_data['status']
X.drop('Marital_Status',axis=1,inplace=True)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.22)

# Model Building
num_trees = 100
max_features = 3
model_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
model_rf.fit(X_train,Y_train)

pred=model_rf.predict(X_test)

# model validataion
# cross val score
kfold= KFold(n_splits=10)
cross_val_score(model_rf,X,Y,cv=kfold).mean()*100
