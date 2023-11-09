import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Visualizar los datos
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data[0], data[1], data[2])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Gráfico de Dispersión en 3D')
plt.show()


# Classificar los datos en 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

print(kmeans.labels_)



