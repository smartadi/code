%% Rotating system Test
clc;
clear all;
close all;

% Construct a rotating eigval and eigvec system
A = eye(3)
C = diag(rand(3,1))

A=C
T = quat2dcm(randrot)

T'*A*T
[V0 D0] = eig(C)
[V1 D1] = eig(A*T)
[V2 D2] = eig(T*A)


%%
n = 10;
A = rand(n,n);
Q = orth(A);

B = eye(n)*diag(rand(n,1));

C = Q'*B*Q;

[V D] = eig(C)


norm(Q'*Q)