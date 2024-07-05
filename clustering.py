import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Load the dataset
data = pd.read_csv('/home/georgikatevski/AI/WeatherCast/weather_classification_data.csv')

# Display the first few rows of the dataframe
print(data.head())  

# Handle missing values
data = data.dropna()

# Encode non-numeric columns
label_encoders = {}
for col in ['Cloud Cover', 'Season', 'Weather Type', 'Location']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
print(data.head())

X = data

y = data['Season']

cols = X.columns

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()

X = ms.fit_transform(X)

X = pd.DataFrame(X, columns=[cols])

X.head()

# Apply PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Combine PCA result with the target column for plotting
finalDf = pd.concat([principalDf, y], axis=1)

# KMeans clustering with PCA-reduced data
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(principalComponents)
labels = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(finalDf['principal component 1'], finalDf['principal component 2'], c=labels, cmap='viridis', marker='o')
plt.title('K-means Clustering with 2 Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.savefig('/home/georgikatevski/AI/WeatherCast/pca_kmeans_clusters.png')
plt.close()

# Check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'.format(correct_labels / float(y.size)))

# Elbow method for determining the optimal number of clusters
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(principalComponents)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.savefig('/home/georgikatevski/AI/WeatherCast/elbow_method.png')
plt.close()

# Repeat clustering for different numbers of clusters
for n_clusters in [2, 3, 4]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(principalComponents)
    labels = kmeans.labels_
    correct_labels = sum(y == labels)
    # Calculate and display descriptive statistics for each cluster
    cluster_stats = finalDf.describe()
    print(f"Descriptive statistics for {n_clusters} clusters:")
    print(cluster_stats)
    print("Result for n_clusters=%d: %d out of %d samples were correctly labeled." % (n_clusters, correct_labels, y.size))
    print('Accuracy score: {0:0.2f}'.format(correct_labels / float(y.size)))
