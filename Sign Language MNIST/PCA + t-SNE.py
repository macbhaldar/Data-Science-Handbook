%matplotlib inline
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.manifold import TSNE
import seaborn as sns

train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
train.head()

# Split data
y = train['label']
X = train.drop(['label'], axis=1)

X.shape, y.shape

# PCA dimensionality reduction
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(X/255)
print('shape of pca_reduced.shape = ', pca_data.shape)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
pca_data = np.vstack((pca_data.T, y)).T
pca_df = pd.DataFrame(pca_data, columns=('PC 1', 'PC 2', 'label'))
pca_df.head()

sns.FacetGrid(pca_df, hue='label', height=8).map(plt.scatter, 'PC 1', "PC 2").add_legend()

# t - SNE dimensionality reduction
model = TSNE(n_components =2, random_state =0, perplexity =50, n_iter=1000)
tsne_data = model.fit_transform(X/255)

print('shape of tsne_reduced.shape = ', tsne_data.shape)

tsne_data = np.vstack((tsne_data.T, y)).T
tsne_df = pd.DataFrame(tsne_data, columns = ('PC 1', 'PC 2', 'label'))
tsne_df.head()

sns.FacetGrid(tsne_df, hue='label', height=8).map(plt.scatter, 'PC 1', "PC 2").add_legend()

# PCA + t-SNE
only_pca_model = decomposition.PCA()
only_pca_model.n_components = 200
only_pca_data = only_pca_model.fit_transform(X/255)
only_pca_data = np.vstack((only_pca_data.T, y)).T

pca_tsne_model = TSNE(n_components =2, random_state =0, perplexity =50, n_iter=1000, verbose=1)
pca_tsne_data = model.fit_transform(only_pca_data)

pca_tsne_data = np.vstack((pca_tsne_data.T, y)).T
pca_tsne_df = pd.DataFrame(pca_tsne_data, columns = ('PC 1', 'PC 2', 'label'))
pca_tsne_df.head()

sns.FacetGrid(pca_tsne_df, hue='label', height=8).map(plt.scatter, 'PC 1', "PC 2").add_legend()

