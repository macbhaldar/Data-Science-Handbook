#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

#loading the data
sales_data= pd.read_csv('Company_Data.csv')
sales_data

sales_data[sales_data['Sales']>=11].head(15)

sales_data.info()

sales_data.describe()

sales_data.columns

for i in sales_data.columns:
    sns.displot(x=i,data = sales_data,color='#F08080')
    plt.title("Distribution of {}".format(i))
    
plt.title('Age vs Sales')
plt.xlabel("AGE",)
sns.scatterplot(x='Age',y='Sales',data=sales_data);

plt.title('Advertising vs Sales')
plt.xlabel("Advertising",)
sns.scatterplot(x='Advertising',y='Sales',data=sales_data);

def get_categorical_data(X):
    sales_status=[]
    for i in X:
        if i>=11:
            sales_status.append('1')
        elif (i>=6):
            sales_status.append('2')
        elif i<6:
            sales_status.append('3')
    return sales_status

x= sales_data['Sales']
sales_data['sales_status']= get_categorical_data(x)
sales_data.sales_status.value_counts()

X=sales_data.iloc[:,1:11]
Y=sales_data.iloc[:,11]

# label encoding for ShelveLoc,Urban and us columns
le=LabelEncoder()
X['ShelveLoc'] =le.fit_transform(X['ShelveLoc'])
X['Urban']= le.fit_transform(X['Urban'])
X['US']= le.fit_transform(X['US'])

# splitting into train and test 
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.15,random_state=44)

# Building a tree model with entropy
model = DecisionTreeClassifier(criterion='entropy',max_depth=10)
model.fit(X_train,Y_train)

fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,15), dpi=300)
plt.title("Tree Visualized")
tree.plot_tree(model,filled=True,fontsize=4)
plt.show();

preds=model.predict(X_test)
preds

pd.Series(preds).value_counts()

pd.crosstab(Y_test,preds)

print("The test data accuracy is :" +str(np.mean(preds==Y_test)))
# The test data accuracy is :0.7833333333333333

pred1=model.predict(X_train)
pd.Series(pred1).value_counts()

pd.crosstab(pred1,Y_train)

print("The training data accuracy is :" + str(np.mean(pred1==Y_train)))
# The training data accuracy is :0.9705882352941176

# Building a model with gini index
model_gini = DecisionTreeClassifier(criterion='gini',max_depth=10)
model_gini.fit(X_train,Y_train)

fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize = (25,15), dpi=200)
features=X_train.columns
tree.plot_tree(model_gini,filled=True,fontsize=5,feature_names=features,class_names=['1','2','3']);

# Model Validation
pred_on_train_data=model_gini.predict(X_train)
pd.Series(pred_on_train_data).value_counts()

pd.crosstab(pred_on_train_data,Y_train)

print("The training data accuracy is :" + str(np.mean(pred_on_train_data==Y_train)))
$ The training data accuracy is :0.9852941176470589
  
pred_on_test_data= model_gini.predict(X_test)
pd.Series(pred_on_test_data).value_counts()

pd.crosstab(pred_on_test_data,Y_test)

print("The test data accuracy is :" + str(np.mean(pred_on_test_data==Y_test)))
# The test data accuracy is :0.7166666666666667
