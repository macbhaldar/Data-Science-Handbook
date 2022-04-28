# The goal is to Build a prediction model for Salary hike
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#loading data
salary_df=pd.read_csv('Salary_Data.csv')
salary_df.head()

# EDA
salary_df.info()
salary_df.describe()

plt.figure(figsize=(6,5))
sns.scatterplot(data=salary_df,x='YearsExperience',y='Salary')
plt.title('Experiance vs salary');
salary_df.corr()
sns.distplot(salary_df['Salary'],color='brown');
sns.distplot(salary_df.YearsExperience)
sns.regplot(x='YearsExperience',y='Salary',data=salary_df,color='green');

# splitting the datset
X=np.array(salary_df['YearsExperience']).reshape(-1,1)
Y=np.array(salary_df['Salary']).reshape(-1,1)
Model = LinearRegression()
Model.fit(X,Y)
## LinearRegression()

predicted=Model.predict(X)

# calculating Mean absolute error
from sklearn.metrics import mean_absolute_error
MAE=metrics.mean_absolute_error(Y,predicted)
print("Mean absolute error is {}".format(MAE))
## Mean absolute error is 4644.2012894435375

from sklearn.metrics import r2_score
Rsquare= r2_score(Y,predicted)
print("The Rsquare value is {}".format(Rsquare))
## The Rsquare value is 0.9569566641435086

print("The intercept value is {}".format(Model.intercept_))
## The intercept value is [25792.20019867]

print("The slope value is{}".format(Model.coef_))
## The slope value is[[9449.96232146]]

# Using statsmodel for calculations

import statsmodels.formula.api as smf
model_smf=smf.ols('Salary~YearsExperience',data=salary_df).fit()
print(model_smf.summary())

# t and p values
model_smf.tvalues,model_smf.pvalues

# predictions
new_data=pd.Series([1.5,9,5,4,2])
pdata=pd.DataFrame(new_data,columns=["YearsExperiance"])
pdata

Model.predict(pdata)

# Using log transformations
x_log=np.log(X)
y_log=np.log(Y)
model1=LinearRegression()
model1.fit(x_log,y_log)

log_pred=model1.predict(x_log)
log_r2score=r2_score(y_log,log_pred)
print("The rsquare value after transforminh the variables into log is {}".format(log_r2score))
## The rsquare value after transforminh the variables into log is 0.9052150725817151

x_sq=X*X
y_sq=Y*Y
model2=LinearRegression()
model2.fit(x_sq,y_sq)
## LinearRegression()

sq_pred=model2.predict(x_sq)
sq_r2core=r2_score(y_sq,sq_pred)

print("The rsquare value after transforminh the variables into squares is {}".format(sq_r2core))

# Using square root
x_sqrt=np.sqrt(X)
y_sqrt=np.sqrt(Y)
model3=LinearRegression()
model3.fit(x_sqrt,y_sqrt)

sqrt_pred=model3.predict(x_sqrt)
sqrt_r2core=r2_score(y_sqrt,sqrt_pred)
print("The rsquare value after transforminh the variables into squares is {}".format(sqrt_r2core))

pd.DataFrame({"models":['model','model(Log)','model(squre)','model(squareroot)'],"rsquare value":[Rsquare,log_r2score,sq_r2core,sqrt_r2core]})
