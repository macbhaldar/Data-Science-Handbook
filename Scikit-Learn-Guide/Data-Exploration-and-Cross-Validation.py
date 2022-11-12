import pandas as pd

adult_census = pd.read_csv("adult-census.csv")
adult_census.head()

target_column = "income"
adult_census[target_column].value_counts()

numerical_columns = [
    "age", "education.num", "capital.gain", "capital.loss",
    "hours.per.week"]
categorical_columns = [
    "workclass", "education", "marital.status", "occupation",
    "relationship", "race", "sex", "native.country"]
all_columns = numerical_columns + categorical_columns + [target_column]

adult_census = adult_census[all_columns]

print(f"The dataset contains {adult_census.shape[0]} samples and "
      f"{adult_census.shape[1]} columns")
      
print(f"The dataset contains {adult_census.shape[1] - 1} features.")

_ = adult_census.hist(figsize=(20, 14))

adult_census["sex"].value_counts()

adult_census["education"].value_counts()

# relationship between "education" and "education-num".
pd.crosstab(index=adult_census["education"], columns=adult_census["education.num"])

# plots on the off-diagonal can reveal interesting interactions between variables.
import seaborn as sns
n_samples_to_plot = 5000
columns = ["age", "education.num", "hours.per.week"]
_ = sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)

_ = sns.scatterplot(
    x="age",
    y="hours.per.week",
    data=adult_census[:n_samples_to_plot],
    hue="income",
    alpha=0.5,
)

import matplotlib.pyplot as plt

ax = sns.scatterplot(
    x="age",
    y="hours.per.week",
    data=adult_census[:n_samples_to_plot],
    hue="income",
    alpha=0.5,
)

age_limit = 27
plt.axvline(x=age_limit, ymin=0, ymax=1, color="black", linestyle="--")

hours_per_week_limit = 40
plt.axhline(y=hours_per_week_limit, xmin=0.18, xmax=1, color="black", linestyle="--")

plt.annotate("<=50K", (17, 25), rotation=90, fontsize=35)
plt.annotate("<=50K", (35, 20), fontsize=35)
_ = plt.annotate("???", (45, 60), fontsize=35)


target_name = "income"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

numerical_columns = ["age", "capital.gain", "capital.loss", "hours.per.week"]
data_numeric = data[numerical_columns]

# create a model using the make_pipeline tool to chain the preprocessing 
# and the estimator in every iteration of the cross-validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())

# %%time
from sklearn.model_selection import cross_validate

model = make_pipeline(StandardScaler(), LogisticRegression())
cv_result = cross_validate(model, data_numeric, target, cv=5)
cv_result

scores = cv_result["test_score"]
print(
    "The mean cross-validation accuracy is: "
    f"{scores.mean():.4f} +/- {scores.std():.4f}"
)

data, target = adult_census.drop(columns="income"), adult_census["income"]
data.head()

target

data.dtypes
data.dtypes.unique()

numerical_columns = ["age", "capital.gain", "capital.loss", "hours.per.week"]
data[numerical_columns].head()

data["age"].describe()

data_numeric = data[numerical_columns]

# train-test split the dataset
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25)
    
print(f"Number of samples in testing: {data_test.shape[0]} => "
      f"{data_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")

print(f"Number of samples in training: {data_train.shape[0]} => "
      f"{data_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
      f" original set")      
      
# create a logistic regression model
from sklearn import set_config
set_config(display='diagram')
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(data_train, target_train)

# Accuracy
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.4f}")
