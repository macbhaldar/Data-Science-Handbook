# Simple Linear Regression
## The goal is to create a model that predicts delivery time using sorting time.

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

# loading data
delivery_df=pd.read_csv('delivery_time.csv')
delivery_df.head()

# EDA
delivery_df.info()

delivery_df.describe()
delivery_df.isna().sum()

sns.set(style='white')
plt.figure(figsize=(6,5))
sns.scatterplot(data=delivery_df,x='Sorting Time',y='Delivery Time')
plt.title('Sorting time vs Delivery time');

sns.distplot(delivery_df['Delivery Time'],color='red')
plt.title('Distribution of Delivery time');

sns.distplot(delivery_df['Sorting Time'],color='Green')
plt.title("Distribution of Sorting time");

delivery_df.corr()

sns.regplot(x='Sorting Time',y='Delivery Time',data=delivery_df,color='brown');
plt.title("Regression line")

# splitting the dataset 
X= np.array(delivery_df['Sorting Time']).reshape(-1, 1)
Y= np.array(delivery_df['Delivery Time']).reshape(-1, 1)
Model Building
model= LinearRegression()

# training the model
model.fit(X,Y)

# model evaluation
predicted=model.predict(X)

# calculating Mean absolute error
from sklearn.metrics import mean_absolute_error
MAE=metrics.mean_absolute_error(Y,predicted)
print("Mean absolute error is {}".format(MAE))

from sklearn.metrics import r2_score
Rsquare= r2_score(Y,predicted)
Rsquare

# Using statsmodel for calculations
delivery_df.rename({"Delivery Time":"Delivery_time","Sorting Time":"Sorting_time"},axis=1,inplace=True)
delivery_df.head()

import statsmodels.formula.api as smf
model1=smf.ols('Delivery_time~Sorting_time',data=delivery_df).fit()
print(model1.summary())

# t and p values
model1.tvalues,model1.pvalues

# Predictions
new_data=pd.Series([6,10,8])
dpred=pd.DataFrame(new_data,columns=['Sorting time'])
dpred

model.predict(dpred)

# Transformation
delivery_df.head()

# Transforming variables for accuracy
model2 = smf.ols('Delivery_time~np.log(Sorting_time)',data=delivery_df).fit()
model2.params
model2.summary()

print(model2.conf_int(0.01))

pred_on_model2 = model2.predict(pd.DataFrame(delivery_df['Sorting_time']))
pred_on_model2.corr(delivery_df['Sorting_time'])

plt.scatter(x=delivery_df['Sorting_time'],y=delivery_df['Delivery_time'],color='red');
plt.scatter(x=delivery_df['Sorting_time'],y=pred_on_model2,color='green');

model3 = smf.ols('np.log(Delivery_time)~Sorting_time',data=delivery_df).fit()
model3.params
model3.summary()

print(model3.conf_int(0.05))
log_pred=model3.predict(delivery_df.iloc[:,1])
pred_on_model3=np.exp(log_pred)
plt.subplot(1,2,1)
sns.scatterplot(x=delivery_df['Sorting_time'],y=delivery_df['Delivery_time'])
plt.subplot(1,2,2)
sns.scatterplot(x=delivery_df['Sorting_time'],y=pred_on_model3)

# Predicted vs actual values
plt.title("Predicted Vs Actual")
plt.scatter(x=pred_on_model3,y=delivery_df['Delivery_time'],color='violet');plt.xlabel("Predicted");plt.ylabel("Actual")

# Using square
# Quadratic model
delivery_df['Sorting_time_sq']=delivery_df.Sorting_time*delivery_df.Sorting_time
model4=smf.ols('Delivery_time~Sorting_time_sq',data=delivery_df).fit()
model4.params
model4.summary()

# Using squareroot
delivery_df['Sorting_time_sqrt']=np.sqrt(delivery_df['Sorting_time'])

# dropped the squared column
delivery_df.head()

model5=smf.ols('Delivery_time~Sorting_time_sqrt',data=delivery_df).fit()
model5.params
model5.summary()

pred_on_model5=model5.predict(delivery_df['Sorting_time_sqrt'])
pred_on_model5.corr(delivery_df['Delivery_time'])

# Predicted vs actual values
plt.title("Predicted Vs Actual")
plt.scatter(x=pred_on_model5,y=delivery_df['Delivery_time'],color='red');plt.xlabel("Predicted");plt.ylabel("Actual")

pd.DataFrame({"Models":['model1','model2','model3','model4','model5'],"intercept":[model1.params[0],model2.params[0],model3.params[0],model4.params[0],model
