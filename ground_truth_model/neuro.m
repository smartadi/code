function dy = neuro(x)

%  FitzHugh-Nagumo model

dy1 = x(1) - 1/3*x(1)^3 - x(2) + I;
dy1 = eps*(b0 + b1*x(1)-x(2));

dy=[dy1;dy2];

