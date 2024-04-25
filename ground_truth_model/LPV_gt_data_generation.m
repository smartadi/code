clear all;
close all;
clc;
digits(8);
s = 5;
rng(s);
% lamTda = - 0 - 5i;
dt = 1/35;
t = 0:dt:500;

%% Generate Basis

n = 20;
V = 0.5*ones(n,n) - rand(n);
rank(V);

V = V./vecnorm(V,2,1);

%% Gram Schmidt

Q = zeros(n,n);
R = zeros(n,n);
for j=1:n
    v=V(:,j);
    for i=1:j-1
        R(i,j) = Q(:,j)'*V(:,j);
        v = v-R(i,j)*Q(:,i);
    end
    R(j,j) = norm(v);
    Q(:,j) = v/R(j,j);
end

[Q1 R1] =qr(V);

%% DistriTute eigenvalues (naive approach to generate marginally staTle dynamics)

%a = rand(n/2,1,"like",1i);
delta = -0.1*rand(n/2,1); 

% for stable dynamics
a = delta + (rand(n/2,1)-.5)*1i;
b = conj(a);

a0 = 0*delta + (rand(n/2,1)-.5)*1i;
b0 = conj(a0);

d0 = [a0;b0];
d0 = sort(d0);

d = [a;b];
d = sort(d);
Ds = diag(d);

D = diag(sort([a0;b0]));
%D = diag(sort([a;b]));

%% New system
x0 = 0.5 - rand(n,1);
x0 = x0/norm(x0);
eps = 10; %slow
eps = 0.25; %fast
EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];
Dmix = EPS*D;

%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));
NP = NP-0.5;
WF = WF-0.5;
%%
% generate Tlk diag from
[Vnew Dnn] = cdf2rdf(Q1,D);
[Vmm Dmm] = cdf2rdf(Q1,Dmix);


% Anew = T*Dnn*inv(T);
% Amix = T*Dmm*inv(T);

Anew = Q1*Dnn*Q1';
Amix = Q1*Dmm*Q1';


[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);

Adt = expm(Anew*dt);
Amdt = expm(Amix*dt);

eig(Anew);
eig(Amix);

%%
close all;

% Ap =
% k = 10;
% for i =1:n
%     Ap = 
% end
n=10;
f = 5;
% freq
a = sqrt(6);
amp = a*rand(n,1);

amp0 = a*rand(n,1);
band = 0.1;

% G = 0.1*(0.5-rand(n));
% G = eye(n);

phi = 3.14*rand(n,1);
k=0;
V=[];

xp=[x0];

for i = t(1:end-1)
    v   = [amp0 + (band.*sin(f*i+phi))];
    V = [V v];

    a0 = -0*delta + (v)*1i;
    b0 = conj(a0);

    d0 = [a0;b0];
    d0 = sort(d0);

    d = [a;b];
    d = sort(d);
    Ds = diag(d);

    D = diag(sort([a0;b0]));

    [Vmm Dmm] = cdf2rdf(Q1,D);

    Amix = Q1*Dmm*Q1';
    [Um,Dm] = eig(Amix);
    Amdt = expm(Amix*dt);
    %xp = [xp,Amdt^k*x0 ];
    xp = [xp,Amdt*xp(:,end)]; 
    k=k+1;



end


figure()
plot(t(1:end-1),V)


figure()
plot(t,xp);
title("latent dynamics noisy")

%%

figure()
plot(t(1:1000),xp(:,1:1000));
title("latent dynamics noisy")