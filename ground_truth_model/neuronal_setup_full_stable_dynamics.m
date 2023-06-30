clear all;
close all;
clc;

% lambda = - 0 - 5i;


t = 0:0.1:100;

% x0=1;
% phi = 0.1;
% x = x0*exp(lambda*t);
% 
% figure()
% plot(t,x)
%% Generate basis

n = 20;
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
%%
Q1(:,1)'*Q1(:,2);
%% EigV of random symmetric matrix (orthogonal eigenvalues)
% 
A = rand(n);
A2 = A.*A';
% [E2 D2] = eig(A2);
%% Assymetric matrix (non-orthogonal eigenvalues)

% [E D] = eig(A);
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

%% New system

% B = E*D*E'
% 
x0 = 0.5 - rand(n,1);
% x=[];
% for i = t
%     x = [x,expm(B*i)*x0];
% end
% figure()
% plot(t,x)
% title('marginally stable system')
% %% Fast and Slow
% 
% % change E to E2
eps = 10;
EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];
% 
% B = EPS*E2*D*E2';
% 
% % x0 = rand(10,1);
% C = 10*(1-2*rand(n,n));
% x=[];
% for i = t
%  
%     x = [x,expm(B*i)*x0];
%     
% end
% y = C*x;
% figure()
% plot(t,x)
% title('marginally stable system with fast and slow modes')
% 
% 
% figure()
% plot(t,y)
% title('output with fast and slow modes')
%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));
%% Non-Orthogonal modes

% B2 = E2*D*E2';
% B2f = EPS*E*D*E';
% 
% x2=[];
% x2f=[];
% for i = t
%     x2 = [x2,expm(B2*i)*x0];
%     x2f = [x2f,expm(B2f*i)*x0];
% end
% figure()
% plot(t,x);hold on;
% plot(t,x2)
% plot(t,x2f)
% 
% 
% figure()
% plot(t,x2)
% title('marginally stable system')
% 
% 
% figure()
% plot(t,x2f)
% title('marginally stable system with fast and slow modes')


% figure()
% plot(t,y)
% title('output with fast and slow modes')
%% stabilise the dynamics

% delta = -5*rand(n/2,1);
% delta = [delta;delta]; 
% Ds = diag(delta + [a;b]);

% Bs = E*Ds*E';
% Bsf = EPS*E*Ds*E';
% 
% 
% xs=[];
% xsf=[];
% for i = t
%  
%     xs = [xs,expm(Bs*i)*x0];
%     xsf = [xsf,expm(Bsf*i)*x0];
% end
% figure()
% plot(t,x);hold on;
% plot(t,xs)
% plot(t,xsf)
% % diag(Ds)
% Bsf
% Bs
%% Real valued A

% A = 0.5*ones(n,n) - rand(n,n)
% B = rand(n,n);
% CC = [];
% DD = [];
% sys = ss(A,B,CC,DD);
% p = pole(sys);
% p
% for i = 1:n
%      if real(p(i)) >= 0
%         p(i) = p(i) - real(p(i))- 0.01*rand(1);
%      end
% end
% 
% K = place(A,B,p);
% 
% Anew = A-B*K
% 
% [Eo Do] = eig(A);
% [En Dn] = eig(Anew);
% diag(Dn)
% p
%%
% Anewf = EPS*Anew;
% 
% xp=[];
% xpf=[];
% for i = t
%     xp = [xp,expm(Anew*i)*x0];
%     xpf = [xpf,expm(Anewf*i)*x0];
% end
% figure()
% plot(t,xp);
% 
% figure()
% plot(t,xpf);
%% Rotation


% Anewf = EPS*Q1'*Anew*Q1;
% 
% xp=[];
% xpf=[];
% for i = t
%     xp = [xp,expm(Anew*i)*x0];
%     xpf = [xpf,expm(Anewf*i)*x0];
% end
% figure()
% plot(t,xp);
% 
% figure()
% plot(t,xpf);
%% Alternative approach

% Consider Block diagonal form M = [a -b;b a] for each complex conjugate
% pair $\lambda = a \pm ib $  

% generate blk diag from
[Vnew Dnew] = cdf2rdf(Q1,D);

% AA = Vnew*Dnew*Vnew';
%%
% AAA = A*Dnew*inv(A);

Anew = A*Dnew*inv(A);
Amix = A*EPS*Dnew*inv(A);

Anew = Q1*Dnew*inv(Q1);
Amix = EPS*Q1*Dnew*inv(Q1);

% [VV,DD] = eig(AAA);
%%

% AAf = EPS*AAA;
% 
% xp=[];
% xpf=[];
% for i = t
%     xp = [xp,expm(AAA*i)*x0];
%     xpf = [xpf,expm(AAf*i)*x0];
% end
% figure()
% plot(t,xp);
% 
% figure()
% plot(t,xpf);
%%
% diag(DD);
% 
% eig(AAf);
% eig(AAA);

eig(Anew)
eig(Amix)
%%
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



subplot()