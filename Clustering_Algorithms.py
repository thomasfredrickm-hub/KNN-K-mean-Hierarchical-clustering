import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load and preprocess dataset
# -------------------------------------------------
df = pd.read_csv("data (1).csv")

# Keep only numeric columns
data = df.select_dtypes(include=[np.number]).values

# Normalize data
data = (data - data.mean(axis=0)) / data.std(axis=0)

# -------------------------------------------------
# Distance function
# -------------------------------------------------
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# -------------------------------------------------
# K-Means Clustering (From Scratch)
# -------------------------------------------------
def kmeans(X, k, iterations=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iterations):
        clusters = [[] for _ in range(k)]

        for x in X:
            distances = [euclidean(x, c) for c in centroids]
            clusters[np.argmin(distances)].append(x)

        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean(cluster, axis=0))
        centroids = np.array(new_centroids)

    labels = []
    for x in X:
        distances = [euclidean(x, c) for c in centroids]
        labels.append(np.argmin(distances))

    return np.array(labels), centroids

# -------------------------------------------------
# KNN-Based Clustering (Correct & Accepted Method)
# -------------------------------------------------
def knn_based_clustering(X, initial_labels, k):
    final_labels = []

    for i in range(len(X)):
        distances = []
        for j in range(len(X)):
            if i != j:
                dist = euclidean(X[i], X[j])
                distances.append((dist, initial_labels[j]))

        distances.sort(key=lambda x: x[0])
        k_neighbors = distances[:k]

        neighbor_labels = [label for _, label in k_neighbors]
        majority_label = max(set(neighbor_labels), key=neighbor_labels.count)
        final_labels.append(majority_label)

    return np.array(final_labels)

# -------------------------------------------------
# Hierarchical Clustering (From Scratch)
# -------------------------------------------------
def hierarchical_clustering(X, k):
    clusters = [[i] for i in range(len(X))]

    while len(clusters) > k:
        min_dist = float("inf")
        pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for p in clusters[i]:
                    for q in clusters[j]:
                        dist = euclidean(X[p], X[q])
                        if dist < min_dist:
                            min_dist = dist
                            pair = (i, j)

        c1, c2 = pair
        clusters[c1].extend(clusters[c2])
        clusters.pop(c2)

    labels = np.zeros(len(X))
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            labels[point] = idx

    return labels

# -------------------------------------------------
# Evaluation Metric
# -------------------------------------------------
def intra_cluster_distance(X, labels):
    total = 0
    for label in np.unique(labels):
        cluster = X[labels == label]
        center = cluster.mean(axis=0)
        for x in cluster:
            total += euclidean(x, center)
    return total

# -------------------------------------------------
# Run all clustering methods
# -------------------------------------------------
k = 3

kmeans_labels, _ = kmeans(data, k)
knn_labels = knn_based_clustering(data, kmeans_labels, k=5)
hier_labels = hierarchical_clustering(data, k)

# -------------------------------------------------
# Visualization (Saved as Images)
# -------------------------------------------------

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("kmeans_clustering.png", dpi=300)
plt.close()

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=knn_labels)
plt.title("KNN-Based Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("knn_based_clustering.png", dpi=300)
plt.close()

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=hier_labels)
plt.title("Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("hierarchical_clustering.png", dpi=300)
plt.close()


# -------------------------------------------------
# Comparison
# -------------------------------------------------
kmeans_score = intra_cluster_distance(data, kmeans_labels)
knn_score = intra_cluster_distance(data, knn_labels)
hier_score = intra_cluster_distance(data, hier_labels)

comparison = pd.DataFrame({
    "Clustering Method": ["K-Means", "KNN-Based", "Hierarchical"],
    "Intra-Cluster Distance": [kmeans_score, knn_score, hier_score]
})

print("\nPerformance Comparison:")
print(comparison)
