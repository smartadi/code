%% Data Generation:
clear all;
close all;
clc;
s = 5;
rng(s);
% lambda = - 0 - 5i;
digits(8);

% US = double(table2array(readtable("data/data_WF_US_small.csv")));
dt=0.1;
t = 0:dt:1000;

% t = 0:dt:20;

% %% Generate basis

n = 8;
V = 0.5*ones(n,n) - rand(n);
rank(V);

V = V./vecnorm(V,2,1);

V(:,1)'*V(:,2);

% %% Gram Schmidt

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
% %% EigV of random symmetric matrix (orthogonal eigenvalues)
% 
%A = rand(n)+1; % random matrix
% A = rand(n); % random matrix
% A2 = A.*A';  % symmetric matrix
% 
% 
% % A = full(sprand(n,n,1));
% 
% 
% B = 2*(0.5 - rand(n)); % random matrix
% % B = 1+  rand(n); % random matrix
% [P, J] = jordan(A2);
% [P2, J] = jordan(B);
% P = real(P);
% P2 = real(P2);
% %% Distribute eigenvalues (naive approach to generate marginally stable dynamics)

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

D = 1*diag(sort([a0;b0]));


% %% New system

x0 = 0.5 - rand(n,1);

%eps = 100;
eps = 0.5;
% eps = 0.05;

EPS = [eye(n/2),zeros(n/2,n/2);
    zeros(n/2,n/2),1/eps*eye(n/2)];

Dmix = EPS*D;

%%% Mix and Match 
WF = full(sprand(n/2,n,1));
NP = full(sprand(n/2,n,1));


%%% Alternative approach 

% generate blk diag from
[Vnew Dnn] = cdf2rdf(Q1,D);
[Vmm Dmm] = cdf2rdf(Q1,Dmix);
%%% 
% 
% Anew = P*Dnn*inv(P)
% 
% 
% Amix = P*Dmm*inv(P)
% 
% 
% % 
% Anew2 = B*Dnn*inv(B)
% Amix2 = B*Dmm*inv(B)

Anew = Q1*Dnn*inv(Q1)
Amix = Q1*Dmm*inv(Q1)

% Eigen Decomposition

[Un,Dn] = eig(Anew);
[Um,Dm] = eig(Amix);

dt=0.1;
Amdt = expm(Amix*dt)

eig(Anew)
eig(Amix)
% %%
NP = NP-0.5;
WF = WF-0.5;
close all;
xp=[];
xpu=[];
xpf=[];
p=[];
k=1;
for i = t
    

    xp =  [xp,  expm(Anew*i)*x0];
    xpf = [xpf, expm(Amix*i)*x0];
    xpu = [xpu,Q1*expm(Dmm*i)*inv(Q1)*x0];
    p = [p,Q1*expm(Dmm*dt)^k*inv(Q1)*x0];
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

% %% Add high freq noise
close all;
xp=[];
xpf=[];
f = 5;
Cn =[];
Cm =[];
V =[];
f = 10*rand(n,1);
amp = 0.01*rand(n,1);

G = 0.5-rand(n);
 
phi = 1*3.14*rand(n,1);
for i = t
    Cn  = [Cn, Q1*expm(Dnn*i)*inv(Q1)];
    Cm  = [Cm, Q1*expm(Dmm*i)*inv(Q1)];
    v   = amp.*sin(f*i+phi);
    V = [v;V];
    xp  = [xp,  expm(Anew*i)*x0 + Cn*V];
    xpf = [xpf, expm(Amix*i)*x0 + Cm*V];
%     xp  = [xp,  expm(Anew*i)*x0 + G*Cn*V];
%     xpf = [xpf, expm(Amix*i)*x0 + G*Cm*V];
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
%%



disp('loading WF PCA projections')

data = ywf;



[dims, time] = size(data);


%    a = [ 0.999505  0.00984985;-0.09849847  0.96995546];
%    k = [0.1;0.1];
%    c = [1,0];
%    r = [0.01];
%    l = 1; 				% Number of outputs

l=n/2
% transpose for Umap
% data = data./vecnorm(data,2,2);
data_r = data(1:l,1:1000);
data_small = data(1:l,1:2000);
    
    
% %
% %   We take a white noise sequence of 4000 points as (unmeasured) input e.
% %   The simulated output is stored in y. (chol(r) makes cov(e) = r).

%     N = 4000;
%     e = randn(N,l)*chol(r);
%     x0 = [10;-10];
%     y = dlsim(a,k,c,eye(l),e,x0);
    
%   The output signals:

figure()
plot(data_r(1,:));hold on
plot(data_r(2,:));
plot(data_r(3,:));
plot(data_r(4,:));
% plot(data_r(:,5));
title('Outputs 1-5');
    

%   Hit any key


%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
       max_order = 60;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       must be an integer
        i = 2*(max_order)/l;
%

  nn= 8
%   
%     [A,du1,C,du2,K,R,AUX] = subid(data_small,[],i,nn,[],[],1);
%     %[As,du1s,Cs,du2s,Ks,Rs,AUXs] = subid_stable(data_small,[],i,nn,[],[],1);
% 
%            % [Ass,du1ss,Css,du2ss,Kss,Rss,AUXss] = subid_sparse(data_small,[],i,nn,[],[],1);
% 
%     
%     
%     era = [];
%     for n = 1:20
%       [A,B,C,D,K,R] = subid(data_small,[],i,n,[],[],1);
%       [yp,erp] = predic(data_small,[],A,[],C,[],K);
%       era(n,:) = erp;
%     end
%     
% %   Hit any key
% 
% 
% %           
% %   We have now determined the prediction errors for all systems 
% %   from order 1 through 6.
% %   Plotting these often gives a clearer indication of the order:
% %   
%     subplot;
%     bar([1:20],era);title('Prediction error');
%     xlabel('System order');
%     
% %   It now even becomes more clear that the system order is 4.
% %    
% %   Hit any key
% 
% 
% %   
% %   We did find this very useful, so we included the above code in 
% %   a function: allord.  The above result could have been obtained 
% %   with the one line code:
% %   
%     [ersa,erpa] = allord(data_small,[],i,[1:20],AUX);
% 
% %   Hit any key
% 
% 
% 
% %   A last feature we would like to illustrate is that subid (or
% %   the other stochastic identification algorithms) also
% %   can work with another basis.     
% %   The order of the system is then equal to the number of singular
% %   values different from zero (instead of the number of angles
% %   different from 90 degrees):
    AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],i,nn,[],[],1);

 [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],i,nn,AUX,'sv');
    %[Asp,du1sp,Csp,du2sp,Ksp,Rsp,AUXsp] = subid_sparse2(data_small,[],i,nn,[],[],1);


%   The order is still clearly equal to 4.
%   
%   This concludes the stochastic demo.  You should now be ready
%   to use subid on your own data.
%   
%   Note that other time series identification algorithms are:
%   sto_stat, sto_alt and sto_pos.
%   
%   For a demonstration of combined identification, see sta_demo.

%%
% predict future
%            xp_{k+1} = A xp_k + B u_k + K (yp_k - C xp_k - D u_k)
%              yp_k   = C xp_k + D u_k
clc;
close all
T = length(data_small);
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);



for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data_small(:,i) - C*x(:,i) );
    y(:,i) = C*x(:,i);
end


figure()
for i=1:l
    subplot(5,3,i)
    plot(y(i,:));hold on;
    plot(data_small(i,:))
end


figure()
for i=1:n
    subplot(5,4,i)
    plot(x(i,:));hold on;
    plot(xpf(i,1:T));hold on;
end
% x0 = x(:,end);

%% Forecast forward
close all;
T = 5000;
data_f = data(1:l,2000:end);

T = length(data_f);
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);

xs = zeros(n,T);
ys = zeros(l,T);
% data_f = data(1:l,5000:10000);

%x(:,1) = x0;

for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data_f(:,i) - C*x(:,i) );
    y(:,i) = C*x(:,i);
    
    xs(:,i+1) = As*xs(:,i) + Ks *( data_f(:,i) - Cs*xs(:,i) );
    ys(:,i) = Cs*xs(:,i);
end



dd = eig(A);


figure()
for i=1:n
    subplot(5,4,i)
    plot(x(i,1:1000));hold on;
    plot(xpf(i,1:1000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')

figure()
for i=1:n
    subplot(5,4,i)
    plot(xs(i,1:1000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')

figure()
for i=1:l
    subplot(5,4,i)
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(ys(i,1:1000),'b');
    title(num2str(i))
end
sgtitle('Forecast outputs UMAP modes')

figure()
for i=1:l
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(ys(i,1:1000),'g');
    title(num2str(i))
end
%%
close all;
figure()
for i=1:6
    subplot(2,3,i)
    plot(xs(i,1500:1750),'b','Linewidth',2);hold on;
    title(num2str(i))
end
sgtitle('Latent states')

figure()
for i=1:l
    subplot(2,3,i)
    
    plot(ys(i,1500:1750),'g','Linewidth',2);hold on;
    plot(data_f(i,1500:1750),'--k');hold on;
    title(num2str(i))
end
sgtitle('Output modes')
%%
close all
figure()
plot(ys(1,1500:1750),'k','Linewidth',2);hold on;
plot(data_f(1,1500:1750),'--k','Linewidth',1.5);hold on;

plot(ys(2,1500:1750),'g','Linewidth',2);hold on;
plot(data_f(2,1500:1750),'--g','Linewidth',1.5);hold on;

plot(ys(3,1500:1750),'b','Linewidth',2);hold on;
plot(data_f(3,1500:1750),'--b','Linewidth',1.5);hold on;

plot(ys(4,1500:1750),'m','Linewidth',2);hold on;
plot(data_f(4,1500:1750),'--m','Linewidth',1.5);hold on;
% 
% plot(ys(5,1500:1750),'r','Linewidth',2);hold on;
% plot(data_f(5,1500:1750),'--r','Linewidth',1.5);hold on;
legend('SysID','measured')

%% 
close all
figure()
plot(xs(1,1500:1750),'k','Linewidth',2);hold on;

plot(xs(2,1500:1750),'g','Linewidth',2);hold on;

plot(xs(3,1500:1750),'b','Linewidth',2);hold on;

plot(xs(4,1500:1750),'m','Linewidth',2);hold on;

plot(xs(5,1500:1750),'r','Linewidth',2);hold on;
% writematrix(y,'WF_PCA_forecast_10k.csv','Delimiter',',')
% writematrix(x,'WF_PCA_latent_10k.csv','Delimiter',',')
%%
figure()
for i=1:n
    subplot(5,4,i)
    plot(xs(i,4500:5000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:l
    subplot(2,5,i)
    plot(data_f(i,4500:5000),'k','Linewidth',2);hold on;
    plot(ys(i,4500:5000),'r');
    title(num2str(i))
end
sgtitle('Forecast outputs UMAP modes')

%%
dd = eig(A)
ds = eig(As)
figure()
plot(dd,'ob');hold on
plot(ds,'or');


%%
figure()
for i=1:n
    subplot(5,4,i)
    plot(xs(i,4500:5000));hold on;
    plot(xpf(i,4500:5000))
    title(num2str(i))
end
sgtitle('Forecast latent states')
