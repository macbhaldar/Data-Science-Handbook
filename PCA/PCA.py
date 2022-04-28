# Principal Component Analysis

# importing the required libraries
import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale 
import warnings
warnings.filterwarnings("ignore")

# loadiing the data
wine_data= pd.read_csv('wine.csv')
wine_data.head()

# dropping the type feature
wine_data.drop('Type',axis=1,inplace=True)
corr=wine_data.corr()
sns.heatmap(corr,cmap='afmhot')

# standardizing the data
scaled_wine_data=scale(wine_data)
scaled_wine_data

pca=PCA(n_components=13)
pca_values= pca.fit_transform(scaled_wine_data)
var=pca.explained_variance_ratio_
var

var1=np.cumsum(np.round(var,4)*100)
var1

pca.components_

plt.plot(var1,"o-r");

# taking 3 pcs for furter calculations
# plotting between PCA1 and PCA2 
x = pca_values[:,0:1]
y = pca_values[:,1:2]
z = pca_values[:,2:3]
plt.scatter(x,y);

# making a dataframe with pc1,pc2,pc3
final_df=pd.concat([pd.DataFrame(pca_values[:,0:3],columns=['pc1','pc2','pc3'])],axis=1)
final_df

# visualizing the pcs
sns.set_style(style='darkgrid')
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df,palette='afmhot_r');

# Using Clustering Algorithms
## Heirerchal
sns.set_theme(context='notebook',
    style='white',
    palette='deep',
    font='sans-serif',
    font_scale=1,
    color_codes=True,
    rc=None)

# import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
# create dendrogram
plt.figure(figsize=(10,8))
dendrogram = sch.dendrogram(sch.linkage(scaled_wine_data, method='complete'))

HC=AgglomerativeClustering(n_clusters=3)
y=pd.DataFrame(HC.fit_predict(scaled_wine_data),columns=['clustersid'])
y.value_counts()

HC_df=pd.concat([wine_data,y],axis=1)
HC_df

HC_df.groupby('clustersid').mean()

## Kmeans
from sklearn.cluster import KMeans

# calculating and plotting within cluster sum of squares
wcss=[]
for i in range(1,6):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=4)
    kmeans.fit(scaled_wine_data)
    wcss.append(kmeans.inertia_)
    
    
plt.plot(range(1,6),wcss,"o-g")
plt.xlabel("clusters")
plt.ylabel("wcss")
plt.show();

kmeans=KMeans(n_clusters=3,random_state=4).fit(scaled_wine_data)
kmeans.labels_

k_y=pd.DataFrame(kmeans.labels_,columns=['Clusterid'])
k_y

K_df=pd.concat([wine_data,k_y],axis=1)
K_df

# Dbscan
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2.5,min_samples=3).fit(scaled_wine_data)
dbscan.labels_

D_y=pd.DataFrame(dbscan.labels_,columns=['Cluster_ID'])
D_y.value_counts()

D_df=pd.concat([wine_data,D_y],axis=1)
D_df

D_df.groupby('Cluster_ID').mean()
