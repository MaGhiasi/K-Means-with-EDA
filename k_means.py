import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from eda import perform_eda

# Get features after EDA
features = perform_eda("Live.csv")

list_WCSS = []
k_range = range(1, 11)

# k-means loop for different K
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=69)
    kmeans.fit(features)
    list_WCSS.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, list_WCSS, 'b-')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
