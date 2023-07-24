clear all;
close all;
clc;

digits(8);
s = 5;
rng(s);
t = 0:0.1:100;


%% Generate basis

n = 8;
V = 0.5*ones(n,n) - rand(n);
rank(V);

V = V./vecnorm(V,2,1);

V(:,1)'*V(:,2);

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

%% EigV of random symmetric matrix (orthogonal eigenvalues)

A = 0.5-rand(n);
A2 = A.*A';

% Computation time increas
% [P, J] = jordan(A2);
% P = real(P);

B = 2*(0.5 - rand(n)); % random matrix
%% Distribute eigenvalues (naive approach to generate marginally stable dynamics)

%a = rand(n/2,1,"like",1i);
delta = -0.01*rand(n/2,1); 

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
x0 = 0.5 - rand(n,1);

eps = 0.25;
EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];
Dmix = diag(EPS*D);
% D = diag(D)
%% Generate Block diagonal forms
Dmm=[];
for i=1:n/2
    dmm=[];
    for j= 1:n/2
        if i==j
            d = [real(Dmix(i*2-1)),-imag(Dmix(i*2-1));
                imag(Dmix(i*2-1)),real(Dmix(i*2-1))];
        else
            d = zeros(2,2);
        end
        dmm = [dmm,d];
    end
    Dmm = [Dmm;dmm];
end


Dnew=[];
for i=1:n/2
    dn=[];
    for j= 1:n/2
        if i==j
            d = [real(D(i*2-1)),-imag(D(i*2-1));
                imag(D(i*2-1)),real(D(i*2-1))];
        else
            d = zeros(2,2);
        end
        dn = [dn,d];
    end
    Dnew = [Dnew;dn];
end



%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));

%% Real valued A
% generate a block diagonal form for complex eigs 
[Vnew Dnew] = cdf2rdf(Q1,D);

Anew = Q1*Dnew*inv(Q1);
Amix = EPS*Q1*Dnew*inv(Q1);

% non orthogonal eigenvectors cause mixing if not shifted to zero mean
% entries
Anew = A*Dnew*inv(A);
Amix = A*EPS*Dnew*inv(A);

Anew = P*Dnew*inv(P);
Amix = P*Dmm*inv(P);
dt = 0.1;
Anewt = expm(Anew*dt);
Amixt = expm(Amix*dt);

% generate jordan form to get a real decomposition and verify(slow!)
% [Pn Jn] = jordan(Anew);
% [Pm Jm] = jordan(Amix);
%%
close all;
xp=[];
xpf=[];
k=0;
for i = t
%     xp = [xp,expm(Anew*i)*x0];
%     xpf = [xpf,expm(Amix*i)*x0];
     
    xp = [xp,Anewt^k*x0];
    xpf = [xpf,Amixt^k*x0];
    k = k+1;
end
    ywf = WF*xpf;
    ynp = NP*xpf;

figure()
plot(t,xp);
title('x')

figure()
plot(t,xpf);
title('x mixed fast and slow')


figure()
plot(t,ywf);
title('output WF')

figure()
plot(t,ynp);
title('output NP')

%% Alternate transform
% P = A;
%% Add high freq noise
close all;
xp=[];
xpf=[];
f = 5;
Cn =[];
Cm =[];
V =[];
f = 10*rand(n,1);
amp = 0.01*rand(n,1);

phi = 3.14*rand(n,1);
for i = t
    Cn  = [Cn, P*expm(Dnew*i)*inv(P)];
    Cm  = [Cm, P*expm(Dmix*i)*inv(P)];
    v   = amp.*sin(f*i+phi);
    V = [v;V];
    xp  = [xp,expm(Anew*i)*x0 + Cn*V];
    xpf = [xpf,expm(Amix*i)*x0 + Cm*V];
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
xpf=[];
f = 5;
Cn =[];
Cm =[];
V =[];
f = 10*rand(n,1);
amp = 0.01*rand(n,1);

phi = 3.14*rand(n,1);
for i = t
    Cm  = [Cm, P*expm(Dmix*i)*inv(P)];
    v   = amp.*sin(f*i+phi);
    V = [v;V];

%     xpf = [xpf,expm(Um*Dm^(i)*Um)*x0 + Cm*V];
    xpf = [xpf,expm(Amix*i)*x0 + Cm*V];

end
V = reshape(V,n,length(t));
ywf = WF*xpf;
ynp = NP*xpf;


figure()
plot(t,xpf);
title("Verify")

figure()
plot(t,ywf);
title("WF output")

figure()
plot(t,ynp);
title("NP output")

figure()
plot(t,V)
title("noise")