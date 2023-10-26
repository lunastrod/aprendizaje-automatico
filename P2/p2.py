datos = {
    'PREVISIÓN': ['NUBLADO', 'LLUVIOSO', 'NUBLADO', 'LLUVIOSO', 'SOLEADO', 'LLUVIOSO', 'SOLEADO', 'SOLEADO', 'NUBLADO', 'SOLEADO'],
    'TEMPERATURA': ['MEDIA', 'ALTA', 'BAJA', 'MEDIA', 'ALTA', 'BAJA', 'ALTA', 'ALTA', 'BAJA', 'MEDIA'],
    'MAREA': ['BAJA', 'MEDIA', 'MEDIA', 'ALTA', 'ALTA', 'BAJA', 'ALTA', 'ALTA', 'MEDIA', 'BAJA'],
    'VIENTO': ['MEDIO', 'DEBIL', 'FUERTE', 'FUERTE', 'DEBIL', 'MEDIO', 'FUERTE', 'MEDIO', 'FUERTE', 'DEBIL'],
    'PESCAR': ['SI', 'SI', 'NO', 'SI', 'NO', 'SI', 'NO', 'NO', 'NO', 'SI']
}

datos_test = {
    'PREVISIÓN': ['SOLEADO', 'NUBLADO', 'NUBLADO', 'SOLEADO', 'LLUVIOSO', 'LLUVIOSO', 'NUBLADO', 'SOLEADO', 'NUBLADO', 'LLUVIOSO'],
    'TEMPERATURA': ['ALTA', 'BAJA', 'MEDIA', 'ALTA', 'MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'MEDIA', 'BAJA'],
    'MAREA': ['MEDIA', 'MEDIA', 'BAJA', 'ALTA', 'BAJA', 'ALTA', 'MEDIA', 'ALTA', 'BAJA', 'MEDIA'],
    'VIENTO': ['DEBIL', 'MEDIO', 'MEDIO', 'FUERTE', 'MEDIO', 'DEBIL', 'MEDIO', 'DEBIL', 'MEDIO', 'FUERTE'],
    'PESCAR': ['NO', 'NO', 'SI', 'NO', 'SI', 'SI', 'SI', 'NO', 'SI', 'SI']
}

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.DataFrame(datos)
X_train = df.drop('PESCAR', axis=1)
X_train = pd.get_dummies(X_train)
y_train = df['PESCAR']

df_test = pd.DataFrame(datos_test)
X_test = df.drop('PESCAR', axis=1)
X_test = pd.get_dummies(X_train)
y_test = df['PESCAR']

# Crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo
clf.fit(X_train, y_train)

# Evaluar el modelo
accuracy = clf.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy}")

plot_tree(clf, filled=True, feature_names=list(X_train.columns))

# ==============================================================================

data=np.array([ [0.89,0.41,0.69,1],
                [0.41,0.39,0.82,1],
                [0.04,0.61,0.83,0],
                [0.75,0.17,0.29,1],
                [0.15,0.19,0.31,0],
                [0.14,0.09,0.52,1],
                [0.61,0.32,0.33,1],
                [0.25,0.77,0.83,1],
                [0.32,0.23,0.81,1],
                [0.40,0.74,0.56,1],
                [1.26,1.53,1.21,0],
                [1.68,1.05,1.22,0],
                [1.23,1.76,1.33,0],
                [1.46,1.60,1.10,0],
                [1.38,1.86,1.75,1],
                [1.54,1.99,1.75,0],
                [1.99,1.93,1.54,1],
                [1.76,1.41,1.34,0],
                [1.98,1.00,1.83,0],
                [1.23,1.54,1.55,0]])

X = data[:, :-1]
y = data[:, -1]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Crear un clasificador de árbol de decisión
clf = DecisionTreeClassifier()

# Entrenar el modelo
clf.fit(X_train, y_train)
plot_tree(clf, filled=True)
# Evaluar el modelo
accuracy = clf.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy}")

# ==============================================================================

# Cargar los datos desde el archivo data.csv
data = pd.DataFrame(pd.read_csv('data.csv'))
X = data[['X1', 'X2', 'X3']]
y = data['Y']

# Crear y entrenar el modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Elige el número de vecinos apropiado
knn_model.fit(X, y)

# Clasificar el vector x = (2, 1, 3)
nueva_instancia = pd.DataFrame({'X1': [2], 'X2': [1], 'X3': [3]})
clase_predicha = knn_model.predict(nueva_instancia)
print(f"La clase predicha para x = (2, 1, 3) es: {clase_predicha[0]}")

# Separar las clases
clases = data['Y'].unique()
print(f"Clases: {clases}\n")
# Calcular la matriz de covarianza para cada clase
for clase in clases:
    # Filtrar el DataFrame por clase
    subset = data[data['Y'] == clase]
    # Seleccionar solo las columnas de características
    features = subset[['X1', 'X2', 'X3']]
    # Calcular la matriz de covarianza
    cov_matrix = features.cov()
    print(f"Matriz de Covarianza para la Clase {clase}:\n{cov_matrix}\n")

# Configurar la figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Crear una gráfica de dispersión tridimensional
scatter = ax.scatter(data['X1'], data['X2'], data['X3'], c=data['Y'], cmap='viridis')
# Configurar la leyenda
ax.legend(*scatter.legend_elements(), title='Clases (Y)')
# Configurar etiquetas de ejes
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Gráfico de Dispersión Tridimensional')
# Mostrar la gráfica
plt.show()