import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs 
# Generate some random data for demonstration 
data, labels = make_blobs(n_samples=300, centers=3, random_state=42) 
# Use DBSCAN for clustering 
dbscan = DBSCAN(eps=0.5, min_samples=5) 
clusters = dbscan.fit_predict(data) 
# Plot the results 
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis') 
plt.title('DBSCAN Clustering') 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.show() 