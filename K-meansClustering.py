import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 

# Generate random data
data = np.random.rand(1000, 2) * 100

# Initialize and fit KMeans
km = KMeans(n_clusters=3, init="random", n_init=10)  # Added n_init to avoid warnings
km.fit(data)

# Extract cluster centers and labels
centers = km.cluster_centers_
labels = km.labels_

print("Cluster centers: ", centers) # Cluster centers
# Define colors and markers for plotting
colors = ["r", "g", "b"]
markers = ["+", "x", "*"]

# Plot data points with their cluster colors and markers
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][1], color=colors[labels[i]], marker=markers[labels[i]])

# Plot cluster centers
plt.scatter(centers[:, 0], centers[:, 1], marker="s", s=100, color="black", linewidths=5)

plt.show()
