clear;  % Clear workspace
clc;    % Clear command window
close all;  % Close all figures
load_system("solucion.slx");

% Parameters
Ts = 100e-3;
numSimulations = 30;

% Initialize matrices to store data
inputMatrix = [];
outputMatrix = [];

% Loop for simulations
for simIdx = 1:numSimulations
    disp(simIdx)
    % Generate random setpoints within the 10x10 environment
    refx = rand() * 10;
    refy = rand() * 10;

    % Run simulation
    out = sim("solucion", "StopTime", num2str(10));

    % Accumulate input and output values
    inputMatrix = [inputMatrix; out.E_d, out.E_theta];
    outputMatrix = [outputMatrix; out.V, out.W];
end

% Save matrices
save('inputMatrix.mat', 'inputMatrix');
save('outputMatrix.mat', 'outputMatrix');