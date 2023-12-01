clc;
clearvars;
close all;

load('trainingsetkmeans.mat');
figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, 'filled');
title('trainingsetkmeans');
xlabel('X1');
ylabel('X2');
zlabel('X3');

num_grupos=3;
[idx, C] = kmeans(X, num_grupos);

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, idx, 'filled');
title('kmeans');
xlabel('X1');
ylabel('X2');
zlabel('X3');

%=======================
load('trainingsetPCA.mat');
X_norm = zscore(X);
[coeff, score, latent, ~, explained] = pca(X_norm);
cumulative_explained = cumsum(explained);
figure;
plot(cumulative_explained, 'o-');
xlabel('Número de componentes principales');
ylabel('Varianza explicada acumulativa');
num_components = 7
X_reduced = score(:, 1:num_components);

%==========================
load('trainingsetSVM1.mat');
figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, Y, 'filled');
title('trainingsetSVM1');
xlabel('X1');
ylabel('X2');
zlabel('X3');
modelo_svm = fitcsvm(X, Y, 'KernelFunction', 'linear');
predicciones = predict(modelo_svm, X);

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, predicciones, 'filled');
title('SVM1');
xlabel('X1');
ylabel('X2');
zlabel('X3');

load('trainingsetSVM2.mat');

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, Y, 'filled');
title('trainingsetSVM2');
xlabel('X1');
ylabel('X2');
zlabel('X3');
modelo_svm = fitcsvm(X, Y, 'KernelFunction', 'linear');
predicciones = predict(modelo_svm, X);
num_errores = sum(predicciones ~= Y);
disp(['Número de errores en la predicción: ', num2str(num_errores)]);
disp(['Tasa de error: ', num2str(num_errores/length(Y))]);

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, predicciones, 'filled');
title('SVM2');
xlabel('X1');
ylabel('X2');
zlabel('X3');

grado_polinomio = 7;
modelo_svm = fitcsvm(X, Y, 'KernelFunction', 'polynomial', 'PolynomialOrder', grado_polinomio, 'Standardize',true);
predicciones = predict(modelo_svm, X);
num_errores = sum(predicciones ~= Y);

disp(['Grado del polinomio: ', num2str(grado_polinomio)]);
disp(['Número de errores en la predicción: ', num2str(num_errores)]);

figure;
scatter3(X(:, 1), X(:, 2), X(:, 3), 10, predicciones, 'filled');
title('SVM2 polinomio');
xlabel('X1');
ylabel('X2');
zlabel('X3');