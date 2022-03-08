## Classification Models

### 1. Logistic Regression Classification
Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Just like Linear regression assumes that the data follows a linear function, Logistic regression models the data using the sigmoid function. As it happens, a sigmoid function, defined as follows, produces output having those same characteristics and thus ensures that the model output always falls between 0 and 1:
![CalculatingProbability](./Data-Science-Handbook/images/CalculatingProbability.png)
Based on the number of categories, Logistic regression can be classified as:
- Binomial: target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc.
- Multinomial: target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
- Ordinal: it deals with target variables with ordered categories. For example, a test score can be categorized as:“very poor”, “poor”, “good”, “very good”. Here, each category can be given a score like 0, 1, 2, 3.
