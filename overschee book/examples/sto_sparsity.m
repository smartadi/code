%
% Demo file for Stochastic Subspace identification
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%
%
%   Consider a multivariable fourth order system a,k,c,r
%   which is driven by white noise and generates an output y: 
%   
%               x_{k+1} = A x_k + K e_k
%                y_k    = C x_k + e_k
%              cov(e_k) = R
echo off;

disp('loading WF PCA projections')
%data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_UMAP_projections_small.csv')))';
%data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_KPCA_projections_small.csv')));
data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_PCA_projections_small.csv')));


data = data./vecnorm(data,2,1);
[dims, time] = size(data);


%    a = [ 0.999505  0.00984985;-0.09849847  0.96995546];
%    k = [0.1;0.1];
%    c = [1,0];
%    r = [0.01];
%    l = 1; 				% Number of outputs

l=10
% transpose for Umap
data_r = data(1:l,1:10000)';
data_small = data(1:l,1:1000)';
    
data_sp = data(1:l,1:5000)';
    
% %
% %   We take a white noise sequence of 4000 points as (unmeasured) input e.
% %   The simulated output is stored in y. (chol(r) makes cov(e) = r).

%     N = 4000;
%     e = randn(N,l)*chol(r);
%     x0 = [10;-10];
%     y = dlsim(a,k,c,eye(l),e,x0);
    
%   The output signals:

figure()
plot(data_r(:,1));hold on
plot(data_r(:,2));
plot(data_r(:,3));
plot(data_r(:,4));
plot(data_r(:,5));
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

  nn= 20
  
    %[A,du1,C,du2,K,R,AUX] = subid(data_small,[],i,nn,[],[],1);
    %[As,du1s,Cs,du2s,Ks,Rs,AUXs] = subid_stable(data_small,[],i,nn,[],[],1);

           % [Ass,du1ss,Css,du2ss,Kss,Rss,AUXss] = subid_sparse(data_small,[],i,nn,[],[],1);

    
%     
%     era = [];
%     for n = 1:20
%       [A,B,C,D,K,R] = subid(data_small,[],i,n,AUX,[],1);
%       [yp,erp] = predic(data_small,[],A,[],C,[],K);
%       era(n,:) = erp;
%     end
    
%   Hit any key


%           
%   We have now determined the prediction errors for all systems 
%   from order 1 through 6.
%   Plotting these often gives a clearer indication of the order:
%   
%     subplot;
%     bar([1:20],era);title('Prediction error');
%     xlabel('System order');
    
%   It now even becomes more clear that the system order is 4.
%    
%   Hit any key


%   
%   We did find this very useful, so we included the above code in 
%   a function: allord.  The above result could have been obtained 
%   with the one line code:
% %   
%     [ersa,erpa] = allord(data_small,[],i,[1:20],AUX);
% 
% %   Hit any key



%   A last feature we would like to illustrate is that subid (or
%   the other stochastic identification algorithms) also
%   can work with another basis.     
%   The order of the system is then equal to the number of singular
%   values different from zero (instead of the number of angles
%   different from 90 degrees):

    [A,du1,C,du2,K,R] = subid(data_small,[],i,nn,[],'sv',1);
    
    [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],i,nn,[],'sv',1);
    [Asp,du1sp,Csp,du2sp,Ksp,Rsp,AUXsp] = subid_sparse2(data_small,[],i,nn,[],'sv',1);


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
T = 1000;
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);

xsp = zeros(n,T);
ysp = zeros(l,T);



for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data_r(4000+(i-1),:)' - C*x(:,i) );
    y(:,i) = C*x(:,i);
    
    xsp(:,i+1) = Asp*xsp(:,i) + Ksp *( data_r(4000+(i-1),:)' - Csp*xsp(:,i) );
    ysp(:,i) = C*xsp(:,i);
end


figure()
for i=1:l
    subplot(5,3,i)
    plot(y(i,:),'r');hold on;
    plot(ysp(i,:),'b');
    plot(data_r(4000:5000,i),'k')
end
sgtitle('output modes')

figure()
for i=1:n
    subplot(5,4,i)
    plot(x(i,:),'r');hold on;
    plot(xsp(i,:),'b')
end
sgtitle('latent states')

x0 = x(:,end);
xsp0 = xsp(:,end);

%%
% Forecast forward

% T = 5000;
T = 10000;
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);
xsp = zeros(n,T);
ysp = zeros(l,T);
xs = zeros(n,T);
ys = zeros(l,T);
% data_f = data(1:l,5000:10000);
data_f = data(1:l,:);
x(:,1) = x0;
xsp(:,1) = x0;
xs(:,1) = x0;

for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data_f(:,i) - C*x(:,i) );
    y(:,i) = C*x(:,i);
    
    xsp(:,i+1) = Asp*xsp(:,i) + Ksp *( data_f(:,i) - Csp*xsp(:,i) );
    ysp(:,i) = Csp*xsp(:,i);
    
    xs(:,i+1) = As*xs(:,i) + Ks *( data_f(:,i) - Cs*xs(:,i) );
    ys(:,i) = Cs*xs(:,i);
end



dd = eig(A);


figure()
for i=1:n
    subplot(5,4,i)
    plot(x(i,1:1000),'r');hold on;
    plot(xsp(i,1:1000),'b')
    plot(xs(i,1:1000),'g')
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:l
    subplot(5,4,i)
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(y(i,1:1000),'r');
    plot(ysp(i,1:1000),'b');
    plot(ys(i,1:1000),'g');
    title(num2str(i))
end
sgtitle('Forecast outputs UMAP modes')

figure()
for i=1:l
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(y(i,1:1000),'r');
    plot(ysp(i,1:1000),'b')
    plot(ys(i,1:1000),'g')
    title(num2str(i))
end

% writematrix(y,'WF_PCA_forecast_10k.csv','Delimiter',',')
% writematrix(x,'WF_PCA_latent_10k.csv','Delimiter',',')
%%
figure()
for i=1:n
    subplot(5,4,i)
    plot(x(i,4500:5000),'r');hold on;
    plot(xsp(i,4500:5000),'b')
    plot(xs(i,4500:5000),'g')
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:l
    subplot(2,5,i)
    plot(data_f(i,4500:5000),'k','Linewidth',2);hold on;
    plot(y(i,4500:5000),'r');
    plot(ysp(i,4500:5000),'b');
    plot(ys(i,4500:5000),'g')
    title(num2str(i))
end
sgtitle('Forecast outputs UMAP modes')
%%
figure()
for i=1:l
    subplot(2,5,i)
    plot(data_f(i,500:1000),'k','Linewidth',2);hold on;
    plot(y(i,500:1000),'r');
    plot(ysp(i,500:1000),'b');
    plot(ys(i,500:1000),'g')
    title(num2str(i))
end
sgtitle('Forecast outputs modes')

%%
dd = eig(A);
ds = eig(As);
dsp = eig(Asp);


figure()
plot(dd,'ok');hold on
plot(ds,'og');
plot(dsp,'ob');


%%
e = max(abs(real(dd)))-1