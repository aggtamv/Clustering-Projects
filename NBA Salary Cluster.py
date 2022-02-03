# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:55:00 2022

@author: bakel
"""

import pandas as pd
import sklearn
import numpy as np
from sklearn import  preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns

#Dataset construction
df = pd.read_csv('Salary df.csv', sep = ';', header = 0)
df = df[~df.Player.duplicated(keep = 'first')]
df.set_index(['Player'], inplace = True)
df.rename(columns = {'Salary 19/20' : 'Salary'}, inplace = True)
df.drop(['Unnamed: 0', 'Salary 18/19', 'Pos'], axis = 1, inplace = True)

#Data pre-process
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = df
X = scaler.fit_transform(X)
X = sc.fit_transform(X)

#t-SNE dimension reduction
from sklearn.manifold import TSNE

tsne_res = TSNE(n_components= 2, perplexity = 5, n_iter = 1500, early_exaggeration= 100, verbose = 2,
                init = 'pca', learning_rate = 200, random_state = 0).fit_transform(X)

tsne_plot = pd.DataFrame(tsne_res, columns = ['t-SNE1', 't-SNE2'])

sns.scatterplot(x = 't-SNE1', y = 't-SNE2', data = tsne_plot, palette = 'bright')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

#Spectral Clustering
from sklearn.cluster import SpectralClustering

cl = SpectralClustering(n_clusters= 4, affinity = 'nearest_neighbors', assign_labels= 'kmeans',
                        n_init = 20, n_neighbors = 16, verbose = 2, random_state= 0)
res = cl.fit(tsne_res)

labels = cl.labels_

#Clustering plot
tsne_plot['Cluster'] = labels

sns.scatterplot(x = 't-SNE1', y = 't-SNE2', hue = 'Cluster',data = tsne_plot, palette = 'bright')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

df['Cluster'] = labels

#Boxplot for clusters created
sns.pairplot(hue = 'Cluster', palette = 'bright', data = df)

sns.boxplot(x = 'Cluster', y = 'Salary', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Salary boxplot')

sns.boxplot(x = 'Cluster', y = 'PTS', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('PTS boxplot')

sns.boxplot(x = 'Cluster', y = 'Age', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Age boxplot')

sns.boxplot(x = 'Cluster', y = 'TRB', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Rebounds boxplot')

sns.boxplot(x = 'Cluster', y = 'AST', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Assists boxplot')

sns.boxplot(x = 'Cluster', y = 'MP', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Minutes played boxplot')

sns.boxplot(x = 'Cluster', y = 'ORPM', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Offensive Rating boxplot')

sns.boxplot(x = 'Cluster', y = 'WINS', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Win contribution boxplot')

sns.boxplot(x = 'Cluster', y = 'STL', data = df, palette = 'bright', showmeans = True ,meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"8"})
plt.title('Steals boxplot')








