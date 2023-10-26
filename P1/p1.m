data=[1 1 0;
      2 1 0.69;
      3 1 1.1;
      4 1 1.39;
      5 1 1.61;
      6 1 1.79;
      7 1 1.95;
      8 1 2.08;
      9 1 2.2;
      10 1 2.3;
]

% 1.1. Realice un script en el que se obtengan los pesos (𝜔଴, 𝜔ଵ 𝑦 𝜔ଶ) que forman la ecuación
% de la recta que se ajusta a la relación entre los datos de entrada y de salida. Para ello utilice
% funciones de alto nivel de Matlab y adjunte en la memoria el valor de los pesos obtenidos.

X = data(:, 1:2);
Y = data(:, 3);
mdl = fitlm(X, Y);

coefficients = mdl.Coefficients.Estimate;

omega_1 = coefficients(2);
omega_2 = coefficients(3);
omega_3 = coefficients(1);

fprintf('Resultados del ajuste lineal usando fitlm:\n');
fprintf('omega_1 = %.4f\n', omega_1);
fprintf('omega_2 = %.4f\n', omega_2);
fprintf('omega_3 = %.4f\n', omega_3);

% 1.2. Represente en una figura en 3D el conjunto de datos de entrenamiento, así como la
% recta que mejor se ajusta a los datos de acuerdo con los valores de los pesos obtenidos en
% el apartado anterior.

plot3(data(:,1),data(:,2),data(:,3),'b*')
hold on;

ye = predict(mdl,X)
plot3(X(:,1), X(:,2), ye, 'r', 'LineWidth', 2);
grid on;






% Valores dados
omega_0_given = 0.24;

% Valores de omega_1 y omega_2 para el gráfico
omega_1_vals = linspace(-2, 2, 100);
omega_2_vals = linspace(-2, 2, 100);

% Crear una cuadrícula de valores para omega_1 y omega_2
[Omega_1, Omega_2] = meshgrid(omega_1_vals, omega_2_vals);

% Inicializar la matriz de costos
J_vals = zeros(size(Omega_1));

% Calcular el costo para cada combinación de omega_1 y omega_2
for i = 1:numel(omega_1_vals)
    for j = 1:numel(omega_2_vals)
        % Calcular la predicción para cada ejemplo
        predictions = omega_0_given + Omega_1(i, j) * X(:, 1) + Omega_2(i, j) * X(:, 2);

        % Calcular el costo usando MSE
        J_vals(i, j) = 1/(2 * length(Y)) * sum((predictions - Y).^2);
    end
end

% Graficar la función de costo
figure;
surf(Omega_1, Omega_2, J_vals, 'EdgeColor', 'none');
xlabel('\omega_1');
ylabel('\omega_2');
zlabel('Costo');


data = [
    0.89 0.41 0.69 1;
    0.41 0.39 0.82 1;
    0.04 0.61 0.83 0;
    0.75 0.17 0.29 1;
    0.15 0.19 0.31 0;
    0.14 0.09 0.52 1;
    0.61 0.32 0.33 1;
    0.25 0.77 0.83 1;
    0.32 0.23 0.81 1;
    0.40 0.74 0.56 1;
    1.26 1.53 1.21 0;
    1.68 1.05 1.22 0;
    1.23 1.76 1.33 0;
    1.46 1.60 1.10 0;
    1.38 1.86 1.75 1;
    1.54 1.99 1.75 0;
    1.99 1.93 1.54 1;
    1.76 1.41 1.34 0;
    1.98 1.00 1.83 0;
    1.23 1.54 1.55 0
];


% 2.1. Realice un script que construya el modelo de clasificador basado en regresión logística,
% utilizando funciones de alto nivel de Matlab. Adjunte en la memoria los datos del modelo
% generado que se pueden visualizar en la línea de comandos de Matlab.

X = data(:, 1:3);
Y = data(:, 4);

mdl = fitglm(X, Y, 'Distribution', 'binomial')

% Obtener las probabilidades predichas
probabilidades_predichas = predict(mdl, X);

% Obtener la clase predicha (0 o 1)
clase_predicha = round(probabilidades_predichas);

figure;
% Puntos con Y=0 (círculo)
indices_y0 = find(Y == 0 & clase_predicha==0);
scatter3(X(indices_y0, 1), X(indices_y0, 2), X(indices_y0, 3), 'o', 'MarkerEdgeColor', 'b');
hold on;
indices_y0 = find(Y == 0 & clase_predicha==1);
scatter3(X(indices_y0, 1), X(indices_y0, 2), X(indices_y0, 3), 'o', 'MarkerEdgeColor', 'r');

% Puntos con Y=1 (cruz)
indices_y1 = find(Y == 1 & clase_predicha==0);
scatter3(X(indices_y1, 1), X(indices_y1, 2), X(indices_y1, 3), 'rx','MarkerEdgeColor', 'r');
indices_y1 = find(Y == 1 & clase_predicha==1);
scatter3(X(indices_y1, 1), X(indices_y1, 2), X(indices_y1, 3), 'rx','MarkerEdgeColor', 'b');

grid on;
data=[1 1 0;
      2 1 0.69;
      3 1 1.1;
      4 1 1.39;
      5 1 1.61;
      6 1 1.79;
      7 1 1.95;
      8 1 2.08;
      9 1 2.2;
      10 1 2.3;
]

% 1.1. Realice un script en el que se obtengan los pesos (𝜔଴, 𝜔ଵ 𝑦 𝜔ଶ) que forman la ecuación
% de la recta que se ajusta a la relación entre los datos de entrada y de salida. Para ello utilice
% funciones de alto nivel de Matlab y adjunte en la memoria el valor de los pesos obtenidos.

X = data(:, 1:2);
Y = data(:, 3);
mdl = fitlm(X, Y);

coefficients = mdl.Coefficients.Estimate;

omega_1 = coefficients(2);
omega_2 = coefficients(3);
omega_3 = coefficients(1);

fprintf('Resultados del ajuste lineal usando fitlm:\n');
fprintf('omega_1 = %.4f\n', omega_1);
fprintf('omega_2 = %.4f\n', omega_2);
fprintf('omega_3 = %.4f\n', omega_3);

% 1.2. Represente en una figura en 3D el conjunto de datos de entrenamiento, así como la
% recta que mejor se ajusta a los datos de acuerdo con los valores de los pesos obtenidos en
% el apartado anterior.

plot3(data(:,1),data(:,2),data(:,3),'b*')
hold on;

ye = predict(mdl,X)
plot3(X(:,1), X(:,2), ye, 'r', 'LineWidth', 2);
grid on;






% Valores dados
omega_0_given = 0.24;

% Valores de omega_1 y omega_2 para el gráfico
omega_1_vals = linspace(-2, 2, 100);
omega_2_vals = linspace(-2, 2, 100);

% Crear una cuadrícula de valores para omega_1 y omega_2
[Omega_1, Omega_2] = meshgrid(omega_1_vals, omega_2_vals);

% Inicializar la matriz de costos
J_vals = zeros(size(Omega_1));

% Calcular el costo para cada combinación de omega_1 y omega_2
for i = 1:numel(omega_1_vals)
    for j = 1:numel(omega_2_vals)
        % Calcular la predicción para cada ejemplo
        predictions = omega_0_given + Omega_1(i, j) * X(:, 1) + Omega_2(i, j) * X(:, 2);

        % Calcular el costo usando MSE
        J_vals(i, j) = 1/(2 * length(Y)) * sum((predictions - Y).^2);
    end
end

% Graficar la función de costo
figure;
surf(Omega_1, Omega_2, J_vals, 'EdgeColor', 'none');
xlabel('\omega_1');
ylabel('\omega_2');
zlabel('Costo');


data = [
    0.89 0.41 0.69 1;
    0.41 0.39 0.82 1;
    0.04 0.61 0.83 0;
    0.75 0.17 0.29 1;
    0.15 0.19 0.31 0;
    0.14 0.09 0.52 1;
    0.61 0.32 0.33 1;
    0.25 0.77 0.83 1;
    0.32 0.23 0.81 1;
    0.40 0.74 0.56 1;
    1.26 1.53 1.21 0;
    1.68 1.05 1.22 0;
    1.23 1.76 1.33 0;
    1.46 1.60 1.10 0;
    1.38 1.86 1.75 1;
    1.54 1.99 1.75 0;
    1.99 1.93 1.54 1;
    1.76 1.41 1.34 0;
    1.98 1.00 1.83 0;
    1.23 1.54 1.55 0
];


% 2.1. Realice un script que construya el modelo de clasificador basado en regresión logística,
% utilizando funciones de alto nivel de Matlab. Adjunte en la memoria los datos del modelo
% generado que se pueden visualizar en la línea de comandos de Matlab.

X = data(:, 1:3);
Y = data(:, 4);

mdl = fitglm(X, Y, 'Distribution', 'binomial')

% Obtener las probabilidades predichas
probabilidades_predichas = predict(mdl, X);

% Obtener la clase predicha (0 o 1)
clase_predicha = round(probabilidades_predichas);

figure;
% Puntos con Y=0 (círculo)
indices_y0 = find(Y == 0 & clase_predicha==0);
scatter3(X(indices_y0, 1), X(indices_y0, 2), X(indices_y0, 3), 'o', 'MarkerEdgeColor', 'b');
hold on;
indices_y0 = find(Y == 0 & clase_predicha==1);
scatter3(X(indices_y0, 1), X(indices_y0, 2), X(indices_y0, 3), 'o', 'MarkerEdgeColor', 'r');

% Puntos con Y=1 (cruz)
indices_y1 = find(Y == 1 & clase_predicha==0);
scatter3(X(indices_y1, 1), X(indices_y1, 2), X(indices_y1, 3), 'rx','MarkerEdgeColor', 'r');
indices_y1 = find(Y == 1 & clase_predicha==1);
scatter3(X(indices_y1, 1), X(indices_y1, 2), X(indices_y1, 3), 'rx','MarkerEdgeColor', 'b');

grid on;
