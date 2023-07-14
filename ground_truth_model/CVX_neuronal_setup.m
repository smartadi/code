clear all;
close all;
clc;
s = 5;
rng(s);
% lambda = - 0 - 5i;
digits(8);

US = double(table2array(readtable("data/data_WF_US_small.csv")));
dt = 0.1
t = 0:0.1:100;
n=12
% x0=1;
% phi = 0.1;
% x = x0*exp(lambda*t);
% 
% figure()
% plot(t,x)
% %% Generate basis
% 
% n = 20;
%%
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
A = rand(n)+1; % random matrix
A2 = A.*A';  % symmetric matrix

% Generating a similarity transform
[P, J] = jordan(A2);
P = real(P);

%% Distribute eigenvalues (naive approach to generate marginally stable dynamics)

%a = rand(n/2,1,"like",1i);
delta = -0.1*rand(n/2,1); 

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
%eps = 100;
% eps = 0.1;
eps = 0.2;

EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];
Dmix = diag(EPS*D);
D = diag(D)

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
Dmm

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
Dnew
%% New system

x0 = 0.5 - rand(n,1);

%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));


%%

Anew = P*Dnew*inv(P);
Amix = P*Dmm*inv(P);


[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);

dt=0.1;
Amdt = expm(Amix*dt)

eig(Anew)
eig(Amix)
%%
NP = NP-0.5;
WF = WF-0.5;
close all;
xp=[];
xpf=[];
for i = t
    xp = [xp,expm(Anew*i)*x0];
    xpf = [xpf,expm(Amix*i)*x0];
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

%% Eigen Decomposition
[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);


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
    Cm  = [Cm, P*expm(Dmm*i)*inv(P)];
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
%

%%
T = 100;
N = 100;

Q = 0.01*eye(n);
%Q = Um(:,1)*Um(:,1)';
R = 1*eye(n);
I = eye(N);
G = [];
F = [];
QN = kron(I,Q);
RN = kron(I,R);
X0=x0;
for i = 1:N
%     F = [F; Um*Dm^(i)*Um'];
    F = real([F; P*expm(Dmm*dt)^(i)*inv(P)]);
    
    
    cc=[];
    for j = 1:N
        if j<=i
%             c = Um*Dm^(i-j+1)*Um';
                        c = P*expm(Dmm*dt)^(i-j+1)*inv(P);

        else
            c = zeros(n,n);
        end
        cc = [cc,c];
    end
        G = [G;cc];
        
end

    cvx_begin 
        variable B(n*(N))
        minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B) 
        subject to
%         for i = 1:N
%             
%         end
        
    cvx_end
    
X = F*X0 + G*B;    

%%
x = reshape(X,n,N);
b = reshape(B,n,N);

%%
figure()
plot(x');hold on;

figure()
plot(b');hold on;
%%
% us = US(:,1:n);
% P=[];
% lambda = 0.1;%             c = P*expm(Dmm*dt)^(i-j+1)*inv(P);

% L = 0.01;
% cvx_begin 
%         variable B(n*(N))
%         minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B ) 
%         subject to
%         for i = 1:N
%              norm(us * B((i-1)*n+1:i*n),1) <= L;
% %             norm(B((i-1)*n+1:i*n),1) <= L
%         end
%         
% cvx_end
% %%
% X = F*X0 + G*B;    
% 
% 
% xc = reshape(X,n,N);
% bc = reshape(B,n,N);
% figure()
% plot(xc');hold on;
% 
% figure()
% plot(bc');hold on;

% %% Images
% close all;
% Imc = [];
% Im = [];
% for i=1:N
%     Im = [Im,us * b(:,i)];
%     Imc = [Imc,us * bc(:,i)];
% end
% 
% im = reshape(Im(:,1),100,100);
% imc = reshape(Imc(:,1),100,100);
% 
% 
% %
% figure()
% subplot(1,2,1)
% image(im,'CDataMapping','scaled');hold on
% colorbar
% subplot(1,2,2)
% image(imc,'CDataMapping','scaled');hold on
% colorbar
% %
% 
% norm(Im(:,1),1)
%%
Q = Um(:,1:2)*Um(:,1:2)';

eigs(Q)
%% Modal Supression
R = 1*eye(n);
Q = 0.01*Um(:,1:2)*Um(:,1:2)' + 0.01*Um(:,3:4)*Um(:,3:4)' + + 0.01*Um(:,5:6)*Um(:,5:6)'

%Q = 0.01*Um(:,7:8)*Um(:,7:8)' + 0.01*Um(:,9:10)*Um(:,9:10)' + + 0.01*Um(:,11:12)*Um(:,11:12)'

Q = real(P*[1*eye(6),zeros(6,6);
    zeros(6,6),0*eye(6)]*inv(P));
% Q = 1*Um(:,1:2)*Um(:,1:2)'
% Q = 0.01*real(P*eye(n)*inv(P))
QN = kron(I,Q);
RN = kron(I,R);

    cvx_begin 
        variable B(n*(N))
        minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B) 
    cvx_end
    
X = F*X0 + G*B; 
x = reshape(X,n,N);
b = reshape(B,n,N);

%%
close all;
figure()
plot(x');hold on;
title('fast modal suppression')

figure()
plot(b');hold on;



%% Input Smoothness and laser intensity

% R = 0*eye(n);
% %Q = 0.01*Um(:,1:2)*Um(:,1:2)'
% l = 0.01;
% 
% Q = 0*Um(:,1:2)*Um(:,1:2)'
% 
% QN = kron(I,Q);
% RN = kron(I,R);
% 
% delta = 0.00;
% 
%     cvx_begin 
%         variable B(n*(N))
%         minimize ((F*X0 + G*B*l)'*QN*(F*X0 + G*B*l) + B'*RN*B*l^2)
%         subject to
%         for i=1:N-1
%             c = zeros(1,N);
%             c(i) = -1;
%             c(i+1) = 1;
%             M = c*c';
%             B'*M*B*l^2 <= delta; 
%         end
%     cvx_end
%     
% X = F*X0 + G*B; 
% x = reshape(X,n,N);
% b = reshape(B,n,N);
% 
% %
% close all;
% figure()
% plot(x');hold on;
% title('modal suppression')
% 
% figure()
% plot(b');hold on;

