import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

# Cargar los datos desde el archivo CSV usando Pandas
data = pd.read_csv('trainingsetkmeans.csv')
x1= data['X1'].to_numpy()
x2= data['X2'].to_numpy()
x3= data['X3'].to_numpy()

# Visualizar los datos en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Datos de Entrenamiento')
plt.show()

# Clasificar los datos en 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
predicted_labels = kmeans.predict(data)

# Visualizar los datos etiquetados con las clases
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x1, x2, x3, c=predicted_labels, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Datos clasificados')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Calcular la media de distancias euclidianas de cada dato al centroide de su cluster
distances = kmeans.transform(data)
print(f"Distancias Euclidianas:\n{distances}\n")
mean_distances=[0,0,0]
count=[0,0,0]
for i in range(len(distances)):
    mean_distances[predicted_labels[i]] += distances[i][predicted_labels[i]]
    count[predicted_labels[i]] += 1
mean_distances = np.divide(mean_distances, count)
    
print(f"Media de Distancias Euclidianas: {mean_distances}")


# ==============================================================================

data = pd.read_csv('trainingsetPCA.csv')

pca = PCA()
X_pca = pca.fit_transform(data)

# Analizar la Varianza Explicada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Visualizar la Varianza Explicada
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Explicada Acumulativa')
plt.title('Varianza Explicada Acumulativa por Componente Principal')
plt.show()

# Decidir el Número de Componentes
n_components = 7  # Establecer el número deseado de componentes principales

# Aplicar PCA con el número seleccionado de componentes
pca_final = PCA(n_components=n_components)
X_final = pca_final.fit_transform(data)

# Mostrar el conjunto de datos transformado
transformed_data = pd.DataFrame(X_final, columns=[f'PC{i}' for i in range(1, n_components + 1)])
print("Conjunto de Datos Transformado:")
print(transformed_data)

#==============================================================================

# Cargar los datos desde el archivo CSV usando Pandas
data = pd.read_csv('trainingsetSVM1.csv')

# Separar los datos por clases
x1 = data['X1']
x2 = data['X2']
x3 = data['X3']
y = data['Y']

# Visualizar los datos en un gráfico de dispersión
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c=y, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Datos de Entrenamiento')
plt.show()

# Crear el modelo SVM con kernel lineal
svm_model = SVC(kernel='linear')
svm_model.fit(np.column_stack((x1, x2, x3)), y)

coef = svm_model.coef_[0]
intercept = svm_model.intercept_[0]

# Crea una malla de puntos para visualizar el plano de decisión
xx, yy = np.meshgrid(np.linspace(x1.min(), x1.max(), 50), np.linspace(x2.min(), x2.max(), 50))
zz = (-coef[0]*xx - coef[1]*yy - intercept) / coef[2]

# Grafica el plano de decisión
ax.plot_surface(xx, yy, zz, alpha=0.5, color='gray')

plt.show()

# Cargar los datos desde el archivo CSV usando Pandas
data = pd.read_csv('trainingsetSVM2.csv')

# Separar los datos por clases
x1 = data['X1']
x2 = data['X2']
x3 = data['X3']
y = data['Y']

# Visualizar los datos en un gráfico de dispersión
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c=y, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.title('Datos de Entrenamiento')

# Crear el modelo SVM con kernel lineal
svm_model = SVC(kernel='linear')
svm_model.fit(np.column_stack((x1, x2, x3)), y)

coef = svm_model.coef_[0]
intercept = svm_model.intercept_[0]

# Crea una malla de puntos para visualizar el plano de decisión
xx, yy = np.meshgrid(np.linspace(x1.min(), x1.max(), 50), np.linspace(x2.min(), x2.max(), 50))
zz = (-coef[0]*xx - coef[1]*yy - intercept) / coef[2]

# Grafica el plano de decisión
ax.plot_surface(xx, yy, zz, alpha=0.5, color='gray')

plt.show()

print("a")

# Crear el modelo SVM con kernel polinomial de grado 7
svm_model_poly = SVC(kernel='poly', degree=10)
svm_model_poly.fit(np.column_stack((x1, x2, x3)), y)

# Predict values using the trained SVM model
y_pred = svm_model_poly.predict(np.column_stack((x1, x2, x3)))
print(f"Predicciones:\n{y_pred}\n")
print(f"Valores Reales:\n{y}\n")

# Visualizar los datos en un gráfico de dispersión 3D
fig_poly = plt.figure(figsize=(10, 8))
ax_poly = fig_poly.add_subplot(111, projection='3d')
ax_poly.scatter(x1, x2, x3, c=y_pred, cmap='viridis')
ax_poly.set_xlabel('X1')
ax_poly.set_ylabel('X2')
ax_poly.set_zlabel('X3')
plt.title('Datos de Entrenamiento con Kernel Polinomial')
print("b")
plt.show()