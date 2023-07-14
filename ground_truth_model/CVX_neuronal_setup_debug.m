clear all;
close all;
clc;
s = 5;
rng(s);
% lambda = - 0 - 5i;
digits(8);

% US = double(table2array(readtable("data/data_WF_US_small.csv")));
dt=0.1;
t = 0:dt:100;

t = 0:dt:100;

%% Generate basis

n = 4;
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

[Q1, R1] =qr(V);
%% EigV of random symmetric matrix (orthogonal eigenvalues)
% 
A = rand(n)+1; % random matrix
A2 = A.*A';  % symmetric matrix


% A = full(sprand(n,n,1));



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

D = 10*diag(sort([a0;b0]));


%% New system

x0 = 0.5 - rand(n,1);

%eps = 100;
eps = 0.1;
eps = 0.1;

EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];

Dmix = EPS*D;

%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));


%% Alternative approach

% Consider Block diagonal form M = [a -b;b a] for each complex conjugate
% pair $\lambda = a \pm ib $  

% generate blk diag from
[Vnew Dnew] = cdf2rdf(Q1,D);
[Vmm Dmm] = cdf2rdf(Q1,Dmix);

% 
% [Vnew Dnew] = cdf2rdf(Q1,D);
% [Vmm Dmm] = cdf2rdf(Q1,Dmix);


AA = Vnew*Dnew*Vnew';
%% 
% AAA = A*Dnew*inv(A);

% Anew = A*Dnew*inv(A);
% Amix = A*EPS*Dnew*inv(A);
%P = inv(P);
Anew = P*Dnew*inv(P);

% Anew = Q1*Dnew*inv(Q1);

%Amix = Q1*EPS*Dnew*inv(Q1);
%Amix = Q1*Dmm*inv(Q1);

Amix = P*Dmm*inv(P)
% Eigen Decomposition

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
xpu=[];
xpf=[];
p=[];
k=1;
for i = t
    
%     xp = [xp, expm(Anew^i)*x0];
    xp = [xp, expm(Anew*i)*x0];
    xpf = [xpf,expm(Amix*i)*x0];
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
%%
T = 100;
N = 200;

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
%     F = real([F; P*expm(Dmm*dt)^(i)*inv(P)]);
    F = real([F; Amdt^(i)]);

    
    
    cc=[];
    for j = 1:N
        if j<=i
%             c = P*expm(Dmm*dt)^(i-j+1)*inv(P);
            c = Amdt^(i-j+1);
        else
            c = zeros(n,n);
        end
        cc = [cc,c];
    end
        G = real([G;cc]);
        
end

%     cvx_begin 
%         variable B(n*(N))
%         minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B) 
%         subject to
% %         for i = 1:N
% %             
% %         end
%         
%     cvx_end
%     
% X = F*X0 + G*B;    
% 
% %
% x = reshape(X,n,N);
% b = reshape(B,n,N);
% 
% x = real([X0,x]);
% 
% %
% close all;
% figure()
% plot(x(1,:));hold on;
% plot(x(2,:));hold on;
% plot(x(3,:));hold on;
% plot(x(4,:));hold on;
% 
% figure()
% plot(b');hold on;

%%
% % us = US(:,1:n);
% % P=[];
% % lambda = 0.1;
% % L = 0.01;
% % cvx_begin 
% %         variable B(n*(N))
% %         minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B ) 
% %         subject to
% %         for i = 1:N
% %              norm(us * B((i-1)*n+1:i*n),1) <= L;
% % %             norm(B((i-1)*n+1:i*n),1) <= L
% %         end
% %         
% % cvx_end
% % %%
% % X = F*X0 + G*B;    
% % 
% % 
% % xc = reshape(X,n,N);
% % bc = reshape(B,n,N);
% % figure()
% % plot(xc');hold on;
% % 
% % figure()
% % plot(bc');hold on;
% 
% % %% Images
% % close all;
% % Imc = [];
% % Im = [];
% % for i=1:N
% %     Im = [Im,us * b(:,i)];
% %     Imc = [Imc,us * bc(:,i)];
% % end
% % 
% % im = reshape(Im(:,1),100,100);
% % imc = reshape(Imc(:,1),100,100);
% % 
% % 
% % %
% % figure()
% % subplot(1,2,1)
% % image(im,'CDataMapping','scaled');hold on
% % colorbar
% % subplot(1,2,2)
% % image(imc,'CDataMapping','scaled');hold on
% % colorbar
% % %
% % 
% % norm(Im(:,1),1)
% %%
% Q = Um(:,1:2)*Um(:,1:2)';
% 
% eigs(Q)
% %% Modal Supression
% R = 1*eye(n);
% %Q = 0.01*Um(:,1:2)*Um(:,1:2)'
% 
% 
% Q = 1*Um(:,1:2)*Um(:,1:2)'
% 
% QN = kron(I,Q);
% RN = kron(I,R);
% 
%     cvx_begin 
%         variable B(n*(N))
%         minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B) 
%     cvx_end
%     
% X = F*X0 + G*B; 
% x = reshape(X,n,N);
% b = reshape(B,n,N);
% 
% %
% figure()
% plot(x');hold on;
% title('modal suppression')
% 
% figure()
% plot(b');hold on;
% %% Tests
% i= 10
% norm(F((i-1)*20+1: i*20 , :) - Um*Dm^(i)*Um',2)
% 
% 
% j = 5
% norm(G((i-1)*20+1: i*20,(j-1)*20+1: j*20) - Um*Dm^(i-j+1)*Um',2)
% 
% 
%% Input Smoothness and laser intensity

R = 1*eye(n);
% Q = real(0.01*Um(:,3:4)*Um(:,3:4)')

Q = 0.01*real(P*[0 0 0 0;
     0 0 0 0 ;
     0 0 1 0; 
     0 0 0 1]*inv(P));

l = 1;

% Q = real([1 0 0 0;
%     0 1 0 0 ;
%     0 0 0 0; 
%     0 0 0 0])

%Q = 0*Um(:,1:2)*Um(:,1:2)'
%Q = eye(4)


QN = kron(I,Q);
RN = kron(I,R);
delta = 0.01

X0=x0;

    cvx_begin 
        variable B(n*(N))
        %minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B) 
        minimize ((F*X0 + G*B)'*QN*(F*X0 + G*B) + B'*RN*B)
        subject to
        for i=1:N-1
            c = zeros(1,N);
            c(i) = -1;
            c(i+1) = 1;
            M = c*c';
            B'*M*B*l^2 <= delta; 
        end
    cvx_end
    
X = F*X0 + G*B;
X2 = F*X0 + 0*G*B;
%%
x = reshape(X,n,N);
x = [x0,x];
b2 = reshape(B,n,N);
x2 = reshape(X2,n,N);
x2 = [x0,x2];
x3 = [x0,x3];
%%
close all;
t = 0:dt:dt*N;

figure()
subplot(3,1,1)
plot(t,x2(1,:),'Linewidth',2);hold on;
plot(t,x2(2,:),'Linewidth',2);hold on;
plot(t,x2(3,:),'Linewidth',2)
plot(t,x2(4,:),'Linewidth',2)
ylabel("state")
title('no-modal suppression')

subplot(3,1,2)
plot(t,x(1,:),'Linewidth',2);hold on;
plot(t,x(2,:),'Linewidth',2);hold on;
plot(t,x(3,:),'Linewidth',2)
plot(t,x(4,:),'Linewidth',2)
ylabel("state")
title('fast mode suppression')

subplot(3,1,3)
plot(t,x3(1,:),'Linewidth',2);hold on;
plot(t,x3(2,:),'Linewidth',2);hold on;
plot(t,x3(3,:),'Linewidth',2)
plot(t,x3(4,:),'Linewidth',2)
ylabel("state")
xlabel("time(sec)")
title('slow mode suppression')

figure()
subplot(2,1,2)
plot(t(1:end-1),b);hold on;
ylabel("input")
title('input for slow mode supression')

subplot(2,1,1)
plot(t(1:end-1),b2);hold on;
ylabel("input")
xlabel("time(sec)")
title('input for fast mode supression')

