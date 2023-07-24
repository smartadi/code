clc
clear all
close all
A = [0,-1, 0, 0;
    1, 0,  0,0;
    0, 0, 0, -10
    0, 0, 10, 0];

n=4;
T = rand(n)+1; % random matrix
T2 = T.*T';  % symmetric matrix
[P, J] = jordan(T2);

P = real(P)
A = P*A*inv(P);
% A = inv(P)*A*P;
B = P*[0 0;1 0;0 0;0 1];
% B = inv(P)*[0 0;1 0;0 0;0 1];



Q = [1,0 0,0;
    0,1 0,0;
    0,0 0.00001,0;
    0,0 0,0.00001];

Q = [0.000001,0 0,0;
    0,0.000001 0,0;
    0,0 1,0;
    0,0 0,1];
R = eye(2);
Q = P*Q*inv(P);
% Q = inv(P)*Q*P;


t = 1:0.1:10;

x0 = [0.5,0.5,0.5,0.5];
xx(:,1) = x0;
xx2(:,1) = x0;


c=0;
dt=0.1;
C=eye(4);
sys1 = ss(A,B,C,[]);
sys2 = c2d(sys1,dt);

At = sys2.A;
Bt = sys2.B;
Ct = sys2.C;

[K,S,e] = dlqr(At,Bt,Q,R,[]);

for i=1:1000
    u1(:,i) = -0*K*xx(:,i);
    u2(:,i) = -1*K*xx2(:,i);

    xx(:,i+1) = At*xx(:,i) + Bt*u1(:,i);
    
    xx2(:,i+1) = At*xx2(:,i) + Bt*u2(:,i);

end




figure()
plot(xx(1,:));hold on;
plot(xx(2,:))
plot(xx(3,:))
plot(xx(4,:))
title('Lqr unsolved')

figure()
plot(xx2(1,:));hold on;
plot(xx2(2,:))
plot(xx2(3,:))
plot(xx2(4,:))
title('Lqr solved')

