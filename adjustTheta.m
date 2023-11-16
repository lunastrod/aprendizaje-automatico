clear;  % Clear workspace
clc;    % Clear command window
close all;  % Close all figures

Ts=0.1
refx=5
refy=1
load_system("solucion.slx")
out=sim("solucion","StopTime",num2str(10))

out.E_d;
out.E_theta;
out.V;
out.W;
out.x;
out.y;
out.theta;