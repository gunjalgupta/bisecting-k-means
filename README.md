# Bisecting k-means

The report explains the methodology used to perform bisecting k-means algorithm on a data set containing 8580 text records in sparse format. It is a text data with news records in document-term sparse matrix format where each line represents a document. The train.dat file is a simple CSR sparse matrix containing the features associated with different feature ids and their counts in the input file. Each pair of values within a line represent the term id and its count in that document.

### Methodology used:
1. To start with, I have taken the dataset train.dat and I have extracted the values, indices, and pointers so that I can create a csr_matrix.

2. After creating the CSR matrix of the document data, I have calculated the TF-IDF(Term Frequency - Inverse Document Frequency) which is used to calculate frequency of a word in a document and based on that frequency it decides how important that word is for that document. This internally multiplies how many times a word appears in the document with the inverse document frequency of the words across a set of documents. I have used an inbuilt tfidftransformer from sklearn.feature_extraction to achieve this and used l2 normalization to calculate this.

3. Keeping Curse of dimensionality in mind the dimensions of the achieved sparse matrix with tf-idf scores must be reduced. The document has 126355 columns in it which are reduced to 200 using Truncated SVD algorithm for dimensionality reduction. I used in-built methods for truncated SVD imported from sklearn.decomposition. Truncated SVD works well with tf-idf matrices, so the input matrix is the tf-idf count of the document data. This yields better results with TruncatedSVD.

4. The program also defines some methods to find initial random centroid, and to recalculate centroid which are used k-means algorithm. TO calculate the distances between the clusters, I have used cosine similarity.

5. Bisecting k-means is then performed on the resulting matrix with 200 columns. The pseudo-code for bisecting k-means is-
a. Initialize the list of clusters to accommodate the cluster consisting of all points and list of clusters as empty list and append initial cluster to the list of clusters
b. Loop until the list of clusters contain ‘k’ clusters.
1. Calculate SSE (Sum of Squared Errors) for all the clusters for some number of iterations and return the index of the value with the highest SSE. Call it dropped cluster Index.
2. Perform kmeans with k=2 on the dropped cluster. Now we have generated two clusters from one bigger cluster with the highest SSE and then drop the bigger cluster after new cluster creation.
3. Delete the dropped cluster index from the clusters list and append new clusters to the list of clusters.
c. The inputs required to perform bisecting k-means are matrix with all cluster points, k= number of clusters required and number of iterations that the algorithm should iterate k-means.
d. The output given is the set of labels which indicate the cluster id to which the document belongs to.

6. To calculate efficiency of the clustering technique, I used the Silhouette coefficient metric from sklearn.metrics import silhouette_score. I implemented bisecting k-means for values k=3,5,7,9...21 and then calculated the silhouette coefficient for the same. The silhouette coefficient can range from -1 to 1. A silhouette coefficient of 0 indicates that clusters are significantly overlapping one another, and a silhouette coefficient of 1 indicates clusters are well-separated so value near to 1 is assumed as the optimal k value.

7. The optimal k value labels are put into an external file output.dat for k=7 and plot a graph of the Silhouette coefficient score and k values. The output.dat file is the final output i.e. prediction file for the performed algorithm.

8. The graph achieved after various runs of the program is as follows:

![image](https://user-images.githubusercontent.com/78285301/142564382-17c28803-748e-45d8-80fa-57bc1a0c4517.png)
