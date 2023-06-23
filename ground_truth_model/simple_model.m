clc;
close all;
clear all;

n= 4;
m= 2;
l= 2;
N= 1000;
x= zeros(n,N);
u= zeros(m,N);
y= zeros(l,N);

A = [0,0,1,0;
     0,0,0,1;
     0,0,0,0;
     0,0,0,0];

B = [0 0;0 0;1 0;0 1];


C = [1,0,0,0;
     0,1,0,0];
 
D = [];

dt = 0.01;

sys1 = ss(A,B,C,D);
sys2 = c2d(sys1,dt);

At = sys2.A;
Bt = sys2.B;
Ct = sys2.C;

Qx = 1*eye(2);
Qv = 0.1*eye(2);
Q = [Qx,zeros(2,2);
    zeros(2,2),Qv];
R = 0.1*eye(2);

[K,S,e] = dlqr(At,Bt,Q,R,[]); 



x0 = [1,1,1,0]'; 
x(:,1) = x0; 
for i=1:N
    u(:,i) = -K*x(:,i);
    x(:,i+1) = At*x(:,i) + Bt*u(:,i);
    y(:,i) = Ct*x(:,i);
end


figure()
plot(x(1,:),x(2,:))

%%
[P J] = jordan(At);
P 
J
%%
x0 = [1,1,0,0]';

x = x0; 
t = 0:dt:dt*N;
Cc=[];
U=[];
j = 1;
for i = t
    
    Cc = [Cc, At^(j-1)*Bt];
    u = -K*x(:,end);
    %u = [0;0];
    U = [u;U];
%     x = [x,inv(P)*expm(J*i)*P*x0 + Cc*U];
    x = [x,inv(P)*J^(i)*P*x0 + Cc*U];
    j = j+1;
end

%%
figure()
plot(x(1,:),x(2,:))

P*expm(J*i)*inv(P)*x0