## Principal component analysis
Perform Principal component analysis and perform clustering using first 3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
optimum number of clusters and check whether we have obtained same number of clusters with the original data  (class column we have ignored at the begining who shows it has 3 clusters)

### WHY PCA?
- When there are many input attributes, it is difficult to visualize the data. There is a very famous term ‘Curse of dimensionality in the machine learning domain.
- Basically, it refers to the fact that a higher number of attributes in a dataset adversely affects the accuracy and training time of the machine learning model.
- Principal Component Analysis (PCA) is a way to address this issue and is used for better data visualization and improving accuracy.

### How does PCA work?
- PCA is an unsupervised pre-processing task that is carried out before applying any ML algorithm. PCA is based on “orthogonal linear transformation” which is a mathematical technique to project the attributes of a data set onto a new coordinate system. The attribute which describes the most variance is called the first principal component and is placed at the first coordinate.
- Similarly, the attribute which stands second in describing variance is called a second principal component and so on. In short, the complete dataset can be expressed in terms of principal components. Usually, more than 90% of the variance is explained by two/three principal components.
- Principal component analysis, or PCA, thus converts data from high dimensional space to low dimensional space by selecting the most important attributes that capture maximum information about the dataset.

### Python Implementation:
- To implement PCA in Scikit learn, it is essential to standardize/normalize the data before applying PCA.
- PCA is imported from sklearn.decomposition. We need to select the required number of principal components.
- Usually, n_components is chosen to be 2 for better visualization but it matters and depends on data.
- By the fit and transform method, the attributes are passed.
- The values of principal components can be checked using components_ while the variance explained by each principal component can be calculated using explained_variance_ratio.
