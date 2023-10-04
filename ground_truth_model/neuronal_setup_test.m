% Main script

clear all;
close all;
clc;
s = 5;
rng(s);
% lambda = - 0 - 5i;
digits(8);

% US = double(table2array(readtable("data/data_WF_US_small.csv")));
% dt=0.01;
dt=1/35;
t = 0:dt:100;

% t = 0:dt:20;

%%% Generate basis

n = 8;
V = 0.5*ones(n,n) - rand(n);
rank(V);

V = V./vecnorm(V,2,1);

V(:,1)'*V(:,2);

%%% Gram Schmidt

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

[Q1, R1] =qr(V);
%%% EigV of random symmetric matrix (orthogonal eigenvalues)
% 
%A = rand(n)+1; % random matrix
A = rand(n); % random matrix
A2 = A.*A';  % symmetric matrix


% A = full(sprand(n,n,1));


B = 2*(0.5 - rand(n)); % random matrix
% B = 1+  rand(n); % random matrix
[P, J] = jordan(A2);
[P2, J] = jordan(B);
P = real(P);
P2 = real(P2);
%%% Distribute eigenvalues (naive approach to generate marginally stable dynamics)

%a = rand(n/2,1,"like",1i);

% delta = -0.01*rand(n/2,1);

delta = -0.01*ones(n/2,1);

% delta0 = 0*rand(n/2,1);

eps = 0.1;
% eps = 0.05;

EPS = [eye(n/4),zeros(n/4,n/4);
    zeros(n/4,n/4),1/eps*eye(n/4)];

fr = (rand(n/2,1)-.5)
fr2 = EPS*fr;
a = delta + fr*1i;
a2 = delta + fr2*1i;
b = conj(a);
b2 = conj(a2);

a0 = 0*delta + fr2*1i;
b0 = conj(a0);

d0 = [a0;b0];
d0 = sort(d0);

d = [a;b];
d = sort(d);
Ds = diag(d);


D = 1*diag(sort([a;b]));
Dmix = 1*diag(sort([a2;b2]));
Dmix0 = 1*diag(sort([a0;b0]));
%%% New system

x0 = 0.5 - rand(n,1);

%eps = 100;


%%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));
NP = NP-0.5;
WF = WF-0.5;

%%% Alternative approach 

% generate blk diag from
[Vnew Dnn] = cdf2rdf(Q1,D);
[Vmm Dmm] = cdf2rdf(Q1,Dmix);
[Vmm Dm0] = cdf2rdf(Q1,Dmix0);
%%% 

Anew = P*Dnn*inv(P)


Amix = P*Dmm*inv(P)

% 
% % 
Anew2 = B*Dnn*inv(B)
Amix2 = B*Dmm*inv(B)


Amix0 = B*Dm0*inv(B)

Anew3 = Q1*Dnn*inv(Q1)
Amix3 = Q1*Dmm*inv(Q1)

% Eigen Decomposition

[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);

Amdt = expm(Amix2*dt)
Amdt0 = expm(Amix0*dt)
T=B;
eig(Anew)
eig(Amix)
%%

close all;
xp=[];
xpu=[];
xpf=[];
p=[];
k=1;
for i = t
    

    xp =  [xp,  expm(Anew*i)*x0];
    xpf = [xpf, expm(Amix*i)*x0];
    xpu = [xpu,P*expm(Dmm*i)*inv(P)*x0];
    p = [p,P*expm(Dmm*dt)^k*inv(P)*x0];
    k = k+1;
end
    ywf = WF*xpf;
    ynp = NP*xpf;
    
close all;    
figure()
plot(t,xp);
title("latent dynamics")

figure()
plot(t,xpf);
title("latent dynamics time scaled")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")

figure()
plot(t,xpu);
title("latent dynamics via modes")

figure()
plot(t,p);
title("latent dynamics via modes discrete")

% NP = NP-0.5;
% WF = WF-0.5;


xp=[];
xpu=[];
xpf=[];
p=[];
k=1;
for i = t
    

    xp =  [xp,  expm(Anew2*i)*x0];
    xpf = [xpf, expm(Amix2*i)*x0];
%     xpu = [xpu,B*expm(Dmm*i)*inv(B)*x0];
%     p = [p,B*expm(Dmm*dt)^k*inv(B)*x0];
%     k = k+1;
end
    ywf = WF*xpf;
    ynp = NP*xpf;
  
figure()
plot(t,xp);
title("latent dynamics")

figure()
plot(t,xpf);
title("latent dynamics time scaled")

xp=[];
xpu=[];
xpf=[];
p=[];
k=1;
for i = t
    

    xp =  [xp,  expm(Anew3*i)*x0];
    xpf = [xpf, expm(Amix3*i)*x0];
%     xpu = [xpu,B*expm(Dmm*i)*inv(B)*x0];
%     p = [p,B*expm(Dmm*dt)^k*inv(B)*x0];
%     k = k+1;
end
    ywf = WF*xpf;
    ynp = NP*xpf;
  
figure()
plot(t,xp);
title("latent dynamics")

figure()
plot(t,xpf);
title("latent dynamics time scaled")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")



%% Add high freq noise
close all;
xp=[];
xpf=[];
f = 5;
Cn =[];
Cm =[];
V =[];
f = 1*rand(n,1);
amp = 0.02*rand(n,1);

G = (0.5-rand(n));
G = 0.5*eye(n); 
phi = 10*3.14*rand(n,1);
for i = t
%     Cn  = [Cn, P*expm(Dnn*i)*inv(P)];
%     Cm  = [Cm, P*expm(Dmm*i)*inv(P)];
    Cn  = [Cn, Q1*expm(Dnn*i)*inv(Q1)];
    Cm  = [Cm, Q1*expm(Dmm*i)*inv(Q1)];
    v   = amp.*sin(f*i+phi);
    V = [v;V];
%     xp  = [xp,  expm(Anew*i)*x0 + Cn*V];
%     xpf = [xpf, expm(Amix*i)*x0 + Cm*V];
    xp  = [xp,  expm(Anew3*i)*x0 + G*Cn*V];
    xpf = [xpf, expm(Amix3*i)*x0 + G*Cm*V];
end
V = reshape(V,n,length(t));
ywf = WF*xpf;
ynp = NP*xpf;

figure()
plot(t,xp);
title("latent dynamics")

figure()
plot(t,xpf);
title("latent dynamics time scaled")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")

figure()
plot(t,V)
title("noise")

%% Verification
close all;
xp=[];
xp0=[];
xpf=[];
f = 5;
Cn =[];
Cm =[];
V =[];
f = 10*rand(n,1);
a = 0.002;
amp = a*rand(n,1);

% G = 0.5*(0.5-rand(n));

phi = 3.14*rand(n,1);

G = eye(n);

k=0;
for i = t
%     Cm  = [Cm, expm(Amix*i)];
    Cm  = [Cm, Amdt^k*G];

%     ff = f.*sin(10*i); 
    v   = amp.*sin(f*i+phi);
    V = [v;V];
%     xpf = [xpf,Amdt^k*x0 + G*Cm*V];
    
%     xpf = [xpf,expm(Amix*i)*x0 + G*Cm*V];
    xpf = [xpf,Amdt^k*x0 + Cm*V];

%     xp = [xp,expm(Amix*i)*x0 ];
    xp = [xp,Amdt^k*x0 ];
    xp0 = [xp0,Amdt0^k*x0 ];
%     xp0 = [xp0,expm(Amix0*i)*x0 ];
    


    
    
    
    k=k+1;

end
V = reshape(V,n,length(t));
% ywf = WF*xpf ;
% ynp = NP*xpf ;
ywf = WF*xpf + normrnd(0,0.02,n/2,length(t));
ynp = NP*xpf + normrnd(0,0.02,n/2,length(t));
%
close all;

figure()
plot(t,xpf);
title("noisy")

figure()
plot(t,xp);
title("original")

figure()
plot(t,V)
title("noise")

figure()
plot(t,xp0);
title("marginal")
%%
figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")
%%

% writematrix([t',xpf'],'data/test_gt_dynamics.csv');
% writematrix([t',ywf'],'data/test_gt_WF_dynamics.csv');
% writematrix([t',ynp'],'data/test_gt_NP_dynamics.csv');
% writematrix([t',V'],'data/test_gt_harmonic_noise.csv');
% save("data/test_gt_model.mat","Amdt","T","Dmm","x0");
% writematrix(Um,'data/test_gt_eigvec.csv');
% writematrix(Dm,'data/test_gt_eigval.csv');
% writematrix(x0,'data/test_gt_init.csv');


