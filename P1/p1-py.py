import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

"""
En la siguiente tabla se muestran los datos de entrenamiento compuestos por dos
características de entrada y una salida.
x1 x2 y
1 1 1.56
2 1 1.95
3 1 2.44
4 1 3.05
5 1 3.81
6 1 4.77
7 1 5.96
8 1 7.45
9 1 9.31
10 1 11.64
"""

data=np.array([ [1,1,1.56],
                [2,1,1.95],
                [3,1,2.44],
                [4,1,3.05],
                [5,1,3.81],
                [6,1,4.77],
                [7,1,5.96],
                [8,1,7.45],
                [9,1,9.31],
                [10,1,11.64]])

print(data)

"""
A continuación, resuelva los siguientes apartados:
1.1. Realice un script (o un Jupyter notebook) de Python en el que se obtengan los pesos
(𝜔𝜔0, 𝜔𝜔1 𝑦𝑦 𝜔𝜔2) que forman la ecuación de la recta que se ajusta a la relación entre los datos
de entrada y de salida. Para ello utilice funciones de alto nivel de la librería scikit-learn y
adjunte en la memoria el valor de los pesos obtenidos.
"""

# Se separan los datos de entrada y de salida
X = data[:,0:2]
y = data[:,2]

# Se crea el modelo de regresión lineal
model = LinearRegression().fit(X, y)

# Representación gráfica de los datos
# tiene que ser en 3D
"""
plt.scatter(X[:,0], y, color='blue')
plt.plot(X[:,0], model.predict(X), color='red', linewidth=2)
plt.title('Datos de entrenamiento')
plt.xlabel('x1')
plt.ylabel('y')
plt.show()
"""
ax = plt.axes(projection='3d')
ax.scatter(X[:,0], X[:,1], y, color='blue', marker='x')
ax.plot(X[:,0], X[:,1], model.predict(X), color='red', linewidth=2)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

"""
2.1. Realice un script (o un Jupyter notebook) de Python en el que se represente en 3D la
función de coste a partir de logaritmo de la verosimilitud (asumiendo que el error sigue una
distribución de tipo Gaussiana, de media cero y desviación típica la unidad) para 𝜔𝜔2 = 0, de
tal forma que se aprecie el área donde dicha función presenta los valores máximos.
"""

# Se crea una malla de puntos
w0 = np.linspace(-10, 10, 100)
w1 = np.linspace(-10, 10, 100)
W0, W1 = np.meshgrid(w0, w1)

# Se calcula la función de coste
coste = np.zeros((len(w0), len(w1)))
for i in range(len(w0)):
    for j in range(len(w1)):
        # Se calcula el valor de la función de coste para cada punto de la malla
        # con la formula: 1/2n * sum((y - y_pred)^2)
        #y_pred = W0[i,j] + W1[i,j]*X[:,0]
        # usando la función de sklearn calculamos la predicción
        y_pred = W0[i,j] + W1[i,j]*X[:,0]
        coste[i,j] = 1/(2*len(X)) * np.sum((y_pred - y)**2)


# Se representa la función de coste
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(W0, W1, coste[:,:], cmap='viridis', edgecolor='none')
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('coste')
plt.show()

"""
Se desea construir un modelo basado en regresión logística binaria que sea capaz de
clasificar los datos de la tabla adjunta.
x1 x2 x3 y
0.89 0.41 0.69 +
0.41 0.39 0.82 +
0.04 0.61 0.83 O
0.75 0.17 0.29 +
0.15 0.19 0.31 O
0.14 0.09 0.52 +
0.61 0.32 0.33 +
0.25 0.77 0.83 +
0.32 0.23 0.81 +
0.40 0.74 0.56 +
1.26 1.53 1.21 O
1.68 1.05 1.22 O
1.23 1.76 1.33 O
1.46 1.60 1.10 O
1.38 1.86 1.75 +
1.54 1.99 1.75 O
1.99 1.93 1.54 +
1.76 1.41 1.34 O
1.98 1.00 1.83 O
1.23 1.54 1.55 O
"""
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

print(data)

"""
3.1. Realice un script (o un Jupyter notebook) de Python que construya el modelo de
clasificador basado en regresión logística, utilizando funciones de alto nivel de scikitlearn. Adjunte en la memoria los pesos del modelo generado.
"""

# Se separan los datos de entrada y de salida
X = data[:,0:3]
y = data[:,3]

# Se crea el modelo de regresión logística
model = LogisticRegression().fit(X, y)

# Se calcula el porcentaje de acierto
print("Porcentaje de acierto: ", model.score(X, y))

# Se calculan los pesos del modelo
print("Pesos del modelo: ", model.coef_)
print("Termino independiente: ", model.intercept_)
print()
"""
3.2. Represente en una figura las entradas y la clase a la que pertenece cada uno de los
datos de entrenamiento, así como la predicción de cada una de las entradas utilizando el
modelo creado en el apartado anterior. Nota: utilice distintos marcadores y colores de forma
que se puedan distinguir las clases y el tipo de salida (de los datos de entrenamiento o de la
predicción a partir del modelo). Se recomienda usar la librería Matplotlib.
"""
# Se calcula la predicción del modelo
y_pred = model.predict(X)

# Se representa la clasificación real
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# si el dato es de la clase 0 se representa con una cruz
# si el dato es de la clase 1 se representa con un punto
# si el dato de la predicción es el mismo que el real se representa en azul
# si el dato de la predicción es distinto al real se representa en rojo
for i in range(len(X)):
    if y[i] == 0:
        if y_pred[i] == 0:
            ax.scatter(X[i,0], X[i,1], X[i,2], c='blue', marker='x')
        else:
            ax.scatter(X[i,0], X[i,1], X[i,2], c='red', marker='x')
    else:
        if y_pred[i] == 0:
            ax.scatter(X[i,0], X[i,1], X[i,2], c='red', marker='.')
        else:
            ax.scatter(X[i,0], X[i,1], X[i,2], c='blue', marker='.')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
plt.show()
