import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from eda import perform_eda


features = perform_eda("Live.csv")

# Range of K values to try
k_values = range(1, 11)

# List to store the WCSS for each K
wcss = []

# Calculate WCSS for each K
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, 'bo-')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.show()
