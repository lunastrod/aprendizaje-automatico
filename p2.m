% Definir las variables
trabajo = {'FIJO'; 'TEMPORAL'; 'FIJO'; 'FIJO'; 'TEMPORAL'; 'FIJO'; 'TEMPORAL'; 'TEMPORAL'; 'FIJO'; 'TEMPORAL'};
ingresos = {'ALTOS'; 'MEDIOS'; 'MEDIOS'; 'BAJOS'; 'ALTOS'; 'BAJOS'; 'BAJOS'; 'MEDIOS'; 'MEDIOS'; 'MEDIOS'};
asnef = {'NO';'NO';'SI';'NO';'SI';'NO';'NO';'NO';'NO'; 'NO'};
cantidad = {'ALTA';'MEDIA';'MEDIA';'BAJA';'ALTA';'MEDIA';'BAJA';'MEDIA';'ALTA';'BAJA'};
conceder = {'SI'; 'SI'; 'NO'; 'SI'; 'NO'; 'SI'; 'NO'; 'SI'; 'SI'; 'SI'};

% Crear la tabla
T = table(categorical(trabajo), categorical(ingresos), categorical(asnef), categorical(cantidad), categorical(conceder) ,'VariableNames', {'Trabajo', 'Ingresos', 'Asnef', 'Cantidad', 'Conceder'});

% Mostrar la tabla
disp(T);

% Crear el modelo de árbol de decisión
treeModel = fitctree(T(:,1:4),T.Conceder, 'MinParentSize', 1);

% Visualizar el árbol de decisión
view(treeModel, 'Mode', 'Graph');

res=groupcounts(T,["Conceder","Trabajo"])
res=res.Percent./100
res.*log2(res)

(res(1)+res(2))*Entropy(res(1),res(2)) + (res(3)+res(4))*Entropy(res(3),res(4))

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
];

X = data(:, 1:2);
Y = data(:, 3);

treeModel = fitctree(X,Y, 'MinParentSize', 1);

% Visualizar el árbol de decisión
view(treeModel, 'Mode', 'Graph');

function result = Entropy(n1, n2)
    result = -n1*log2(n1)-n2*log2(n2)
end