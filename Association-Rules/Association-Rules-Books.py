# importing libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data
books_data=pd.read_csv('book.csv')
books_data

# Building apriori algorithm with 10% support and 90% confidence.
frequent_books = apriori(books_data,min_support=0.10,use_colnames=True)
frequent_books

A_R= association_rules(frequent_books,min_threshold=0.9,metric='lift')
A_R

A_R.sort_values('lift',ascending = False)

A_R[A_R.lift>1]

# Visualizing the Rules
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
sns.scatterplot(A_R['confidence'],A_R['lift'])
plt.title("Apriori algorithm results with 10% support and 90% confidence");

# Building apriori algorithm with 15% support and 85% confidence.
frequent_books1=apriori(books_data,min_support=0.15,use_colnames=True)
frequent_books1

A_R1=association_rules(frequent_books1,min_threshold=0.85,metric='lift')
A_R1

A_R1[A_R1.lift>1]

# visualizing the second rule
sns.scatterplot(A_R1['confidence'],A_R1['lift'])
plt.title("Apriori algorithm results with 15% support and 85% confidence");
