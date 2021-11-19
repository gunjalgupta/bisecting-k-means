#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import sys
import numpy
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


#Convert sparse data into usable csr format

with open('train.dat', 'r') as fr:
    lines = fr.readlines()
    nrows = len(lines)
    ncols = 0 
    nnz = 0 
    for i in range(nrows):
        p = lines[i].split()
              nnz += len(p)//2
        for j in range(0, len(p), 2): 
            cid = int(p[j]) - 1
            if cid+1 > ncols:
                ncols = cid+1

value = np.zeros(nnz, dtype=np.single)
index = np.zeros(nnz, dtype=np.intc)
pointer = np.zeros(nrows+1, dtype=np.longlong)
n = 0 
for i in range(nrows):
    p = lines[i].split()
    for j in range(0, len(p), 2): 
        index[n] = int(p[j]) - 1
        value[n] = float(p[j+1])
        n += 1
    pointer[i+1] = n 
    
assert(n == nnz)
    
cm=csr_matrix((value, index, pointer), shape=(nrows, ncols), dtype=np.single)


tfidfTransformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=False)
tfidf = tdidfTransformer.fit_transform(cm)


sparse_matrix = tfidf.toarray()


#Perform Truncated SVD to reduce the number of columns to 200 ->Dimensionality reduction

svd = TruncatedSVD(n_components=200)


X = svd.fit_transform(sparse_matrix)
numpy.set_printoptions(threshold=sys.maxsize)

def initialCentroids(matrix):
    matrixShuffled = shuffle(matrix, random_state=0)
    return matrixShuffled[:2,:]

def similarity(matrix, centroids):
    similarities = matrix.dot(centroids.T)
    return similarities

def findClusters(matrix, centroids):
    
    clusterA = list()
    clusterB = list()
    
    similarityMatrix = similarity(matrix, centroids)
    
    for index in range(similarityMatrix.shape[0]):
        similarityRow = similarityMatrix[index]
        
        #Sort the index of the matrix in ascending order of value and get the index of the last element
        #This index will be the cluster that the row in input matrix will belong to
        similaritySorted = np.argsort(similarityRow)[-1]
        
        if similaritySorted == 0:
            clusterA.append(index)
        else:
            clusterB.append(index)
        
    return clusterA, clusterB



def recalculateCentroid(matrix, clusters):
    centroids = list()
    
    for i in range(0,2):
        cluster = matrix[clusters[i],:]
        clusterMean = cluster.mean(0)
        centroids.append(clusterMean)
        
    centroids_array = np.asarray(centroids)
    
    return centroids_array

# Defining function to perform k-means

def kmeans(matrix, numberOfIterations):
    
    centroids = initialCentroids(matrix)
    
    for _ in range(numberOfIterations):
        
        clusters = list()
        
        clusterA, clusterB = findClusters(matrix, centroids)
        
        if len(clusterA) > 1:
            clusters.append(clusterA)
        if len(clusterB) > 1:
            clusters.append(clusterB)
            
        centroids = recalculateCentroid(matrix, clusters)
        
    return clusterA, clusterB

#Defining function to calculate SSE

def calculateSSE(matrix, clusters):
    
    SSE_list = list()
    SSE_array = []
    
    for cluster in clusters:
        members = matrix[cluster,:]
        SSE = np.sum(np.square(members - np.mean(members)))
        SSE_list.append(SSE)
        
    SSE_array = np.asarray(SSE_list)
    dropClusterIndex = np.argsort(SSE_array)[-1]
            
    return dropClusterIndex

# Defining Bisecting k-means algorithm

def bisecting_kmeans(matrix, k, numberOfIterations):
    
    clusters = list()
    
    initialcluster = list()
    for i in range(matrix.shape[0]):
        initialcluster.append(i)
    
    clusters.append(initialcluster)
    
    while len(clusters) < k:

        dropClusterIndex = calculateSSE(matrix, clusters)
        droppedCluster = clusters[dropClusterIndex]
        
        clusterA, clusterB = kmeans(matrix[droppedCluster,:], numberOfIterations)
        del clusters[dropClusterIndex]
        
        actualClusterA = list()
        actualClusterB = list()
        for index in clusterA:
            actualClusterA.append(droppedCluster[index])
            
        for index in clusterB:
            actualClusterB.append(droppedCluster[index])
        
        clusters.append(actualClusterA)
        clusters.append(actualClusterB)
    
    labels = [0] * matrix.shape[0]

    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels

# Calling the function bisecting k-means
kValues = list()
scores = list()

for k in range(3, 22,2):
    labels = bisecting_kmeans(X, k, 10)
    if (k == 7):
         # write result to output file
        outputFile = open("output.dat", "w")
        for index in labels:
            outputFile.write(str(index) +'\n')
        outputFile.close()

    sscore = silhouette_score(X, labels)
    kValues.append(k)
    scores.append(sscore)
    print ("For K= %d silhouette_coefficient Score is %f" %(k, sscore))

#Plotting the values of silhouette score on graph

plt.plot(kValues, scores)
plt.xticks(kValues, kValues)
plt.xlabel('Number of Clusters k')
plt.ylabel('silhouette_coefficient Score')
plt.title('Trend of Average Distance to Centroid/Diameter')
plt.grid(linestyle='dotted')

plt.savefig('plot.png')
plt.show()

