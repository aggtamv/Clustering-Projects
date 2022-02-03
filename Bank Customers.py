# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:29:41 2022

@author: bakel
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Inserting Data
df = pd.read_csv('Bank.csv', sep  = ',', header = 0)
df.index = df['CUST_ID']
df.drop(['CUST_ID'], axis = 1, inplace = True)
df.isna().sum()
df.dropna(axis = 0, inplace = True)

#Data pre-process
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = df
X = scaler.fit_transform(X)
X = sc.fit_transform(X)

#t-SNE Dimension Reduction
from sklearn.manifold import TSNE

tsne_model = TSNE(init = 'pca', random_state = 0, verbose = 3, n_jobs= -1,
                  learning_rate = 1000, early_exaggeration= 12, perplexity = 30) 
tsne_res = tsne_model.fit_transform(X)

tsne_plot = pd.DataFrame(tsne_res, columns = ['t-SNE1', 't-SNE2'])


sns.scatterplot(x = 't-SNE1', y = 't-SNE2', data = tsne_plot, palette = 'bright')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#Umap Reduction
import umap

umap_model = umap.UMAP(n_neighbors= 30, min_dist= 0.8, n_components = 2, n_jobs = -1, verbose = 3, random_state = 0)
umap_res = umap_model.fit_transform(X)
umap_plot = pd.DataFrame(umap_res, columns = ['Umap1', 'Umap2'])

sns.scatterplot(x = 'Umap1', y = 'Umap2', data = umap_plot, palette = 'bright')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#Spectral Clustering
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters= 6, affinity = 'nearest_neighbors', assign_labels= 'kmeans',
                        n_init = 20, n_neighbors = 300, verbose = 3)

res = sc.fit(tsne_res)

labels = sc.labels_

tsne_plot['Cluster'] = labels

#t-SNE plot
sns.scatterplot(x = 't-SNE1', y = 't-SNE2', hue = 'Cluster',data = tsne_plot, palette = 'bright')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#Eigengap plot for choosing optimal number of clusters
from scipy.spatial.distance import pdist, squareform
def getAffinityMatrix(coordinates, k = 90):
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix
affinity_matrix = getAffinityMatrix(X, k = 30)


import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
def eigenDecomposition(A, plot = True, topK = 90):
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    eigenvalues, eigenvectors = LA.eig(L)
    
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors

affinity_matrix = getAffinityMatrix(X, k = 90)
k, _,  _ = eigenDecomposition(affinity_matrix)
print(f'Optimal number of clusters {k}')


#Boxplots Construction for clustering evaluation
df['Cluster'] = labels

sns.boxplot(x = 'Cluster', y = 'BALANCE', data = df, palette = 'bright', showfliers = False)
df.columns
best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE","CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
best_vals = df[best_cols].iloc[ :, 1:].values
best_cols.append("Cluster")
sns.pairplot(df[ best_cols ], hue="Cluster", palette = 'bright')

describe  = df.describe()
Cluster_0 = df[df['Cluster'] == 0]
sns.pairplot(Cluster_0[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_0)

Cluster_1 = df[df['Cluster'] == 1]
sns.pairplot(Cluster_1[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_1)

Cluster_2 = df[df['Cluster'] == 2]
sns.pairplot(Cluster_2[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_2)

Cluster_3 = df[df['Cluster'] == 3]
sns.pairplot(Cluster_3[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_3)

Cluster_4 = df[df['Cluster'] == 4]
sns.pairplot(Cluster_4[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_4)

Cluster_5 = df[df['Cluster'] == 5]
sns.pairplot(Cluster_5[ best_cols ],hue = 'Cluster', palette = 'bright', data = Cluster_5)

df.describe()

sns.boxplot(x = 'Cluster', y = 'BALANCE', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Balance variation')

sns.boxplot(x = 'Cluster', y = 'PURCHASES', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Purchases variation')

sns.boxplot(x = 'Cluster', y = 'CASH_ADVANCE', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Cash advance variation')

sns.boxplot(x = 'Cluster', y = 'CREDIT_LIMIT', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Credit limit variation')

sns.boxplot(x = 'Cluster', y = 'MINIMUM_PAYMENTS', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Minimum payments variation')

sns.boxplot(x = 'Cluster', y = 'PAYMENTS', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Payments variation')

sns.boxplot(x = 'Cluster', y = 'INSTALLMENTS_PURCHASES', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"},showfliers = False)
plt.title('Installments purchase variation')









