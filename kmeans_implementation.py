#!/usr/bin/env python3
# coding: utf-8

# In[54]:


#All the required imports
import pandas as pd
import urllib
import gzip
import os
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import itertools
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# In[18]:


#To get the current working directory of the user system
local=os.getcwd()

# Reading the csv file
file_df = pd.read_csv(local + '/human_skin_microbiome.csv');


# In[ ]:


#Functions
#function to produce the combinations form a list of alphabets A,C,G,T
def a(val):
    alphabets = ['A','C','G','T']
    keywords = [''.join(i) for i in itertools.product(alphabets, repeat = val)]
    comb = list(keywords)
    return comb

#function to calculate the no of overlapped occurances of a substring in a string
def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count


# In[20]:


#val= int(sys.argv[1])
val = 4
wordtuples = a(val)

#Creating an empty dataframe with string combinations from [A,C,G,T] as the names of the columns, to be used later in the program.
df_ = pd.DataFrame(index=range(0,file_df.shape[0]), columns=wordtuples)
df_ = df_.fillna(0)

#Looping through the dataframe-getting the zip file-then unzipping-performing data preprocessing-and finally storing the cleaned
# data in a new dataframe to be used during clustering
for row in file_df.itertuples():
    index_file_df= row[0]
    url = row[2];
    req = urllib.request.urlretrieve(url,'genome1.gz');
    
    with gzip.open(local + '\genome1.gz', 'rt') as f:
        file_content = f.read()
    
    my_dna = file_content.replace("\n",'|')
    
    new_dna = re.sub(r'\>(.*?)\|', r'', my_dna)
    
    new_dna = new_dna.replace("|",'')
    
    wordfreq = []
    
    for key in wordtuples:
        x = occurrences(new_dna, key);
        wordfreq.append(x)
         
    df_.iloc[index_file_df] = wordfreq

#Adding Organism name column to newly created dataframe    
df_['Organism'] = file_df['Organism']

#Changing the position of Organism column to being first column of dataframe
df_ = df_.reindex(columns=['Organism'] + list(df_.columns[:-1]))


X = np.array(df_.drop(['Organism'], 1))
y = np.array(df_['Organism'])

#Calculating the inertia between the clusters and plotting it to find the best value for k
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    Model = KMeans(n_clusters=k).fit(X)
    Model.fit(X)
    Sum_of_squared_distances.append(Model.inertia_)

# Plot the elbow
plt.plot(K,Sum_of_squared_distances ,'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('The Elbow Method for optimal k')
plt.show()

#Calculating silhouette scores for each cluster and plotting the graph to find the best value for k
K = range(2,10)
silhouette_avg = []
for k in K:
    
    clusterer = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg.append(silhouette_score(X, cluster_labels,metric='euclidean'))
    
#print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
dict_sil = dict(zip(silhouette_avg, K))
     
plt.plot(K,silhouette_avg ,'bx-')
plt.xlabel('k')
plt.ylabel('silhouette_avg')
plt.title('silhouette graph')
plt.show()

max = silhouette_avg[0];
for z in silhouette_avg: 
    if z > max: 
        max = z 
if max > 0:
	print('The best value for K is:', dict_sil[max])

finalModel = KMeans(n_clusters = dict_sil[max]).fit(X)
finalModel.fit(X)
plt.scatter(X[:, 0], X[:, 1], c=finalModel.labels_, s=50, cmap='rainbow')
centers = finalModel.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#Plot showing Cluster Analysis and Dendograms
linked = linkage(X, 'single')
labelList = y
plt.figure(figsize=(10, 7))
plt.title("Dendograms")
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True, get_leaves=True)
plt.xticks(rotation=90)
plt.show()


# In[ ]:




