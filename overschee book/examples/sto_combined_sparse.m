%
% Demo file for Stochastic Subspace identification
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%
clear all;
close all;
clc

%
%   Consider a multivariable fourth order system a,k,c,r
%   which is driven by white noise and generates an output y: 
%   
%               x_{k+1} = A x_k + K e_k
%                y_k    = C x_k + e_k
%              cov(e_k) = R
echo off
disp('loading WF UMAP projections')
data_NP = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/smooth_spike_KPCA_proj_10k.csv')));
%data_NP = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/smooth_spike_PCA_proj_10k.csv')));
%data_NP = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/smooth_spike_UMAP_proj_10k.csv')));

data_WF = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_PCA_projections_small.csv')));
%data_WF = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_KPCA_projections_small.csv')));
%data_WF = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_UMAP_projections_small.csv')))';


%% normalize and take first 10 modes as output

l= 5;

NP = data_NP(1:l,:);
WF = data_WF(1:l,:);

NP_n = NP./vecnorm(NP,2,1);
WF_n = WF./vecnorm(WF,2,1);
%% 
data_np = NP_n(1:l,:)';
data_wf = WF_n(1:l,:)';

data_r = [NP_n(1:l,:)',WF_n(1:l,:)'];

data_small  = data_r(1:2000,:);

data_sp  = data_r(1:5000,:);

l=2*l;
disp('   Hit any key to continue')

[dims, time] = size(data_r);


%    a = [ 0.999505  0.00984985;-0.09849847  0.96995546];
%    k = [0.1;0.1];
%    c = [1,0];
%    r = [0.01];
%    l = 1; 				% Number of outputs

% transpose for Umap
%data_r = data(1:l,1:5000)';
%data_small = data(1:l,1:1000)';
    
    
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
%   Hit any key

%
%   The subspace algorithms is now e asily started.
%   Note the dummy outputs (du1 and du2) where normally the B and D 
%   matrices are located.  Also note that u = [];
%   When prompted for the system order you should enter 4.
%   
    %[A,du1,C,du2,K,R] = subid(data_small,[],i);

%   Did you notice the order was very easy to determine 
%   from the number of principal angles different from 90 degrees?

%   Hit any key
% pause
% clc
% 
% %   Just to make sure we identified the original system again,
% %   we will compare the original and estimated transfer function.
% %   
%     M1 = dbode(A,K,C,eye(l),1,1,w);
%     M2 = dbode(A,K,C,eye(l),1,2,w);
%     figure(1)
%     hold off;subplot;
%     subplot(221);plot(w/(2*pi),[m1(:,1),M1(:,1)]);title('Output 1 -> Output 1');
%     subplot(222);plot(w/(2*pi),[m2(:,1),M2(:,1)]);title('Output 2 -> Output 1');
%     subplot(223);plot(w/(2*pi),[m1(:,2),M1(:,2)]);title('Output 1 -> Output 2');
%     subplot(224);plot(w/(2*pi),[m2(:,2),M2(:,2)]);title('Output 2 -> Output 2');
% 
% %   As you can see, the original and identified spectral factor are close
% %   
% %   Hit any key
% pause
% clc

%   The function "predic" allows you to check the size of the prediction
%   error.  This is a measure for the difference between the original 
%   and the predicted output:
%   
  %  [yp,erp] = predic(data_small,[],A,[],C,[],K);
%
%   erp contains the error per output in percentage:    
%   While yp contains the predicted output: 
%     figure()
%     plot([data_r(100:400,1),yp(100:400,1)]);hold on
%     plot([data_r(100:400,2),yp(100:400,2)])
%     plot([data_r(100:400,3),yp(100:400,3)])
%     plot([data_r(100:400,4),yp(100:400,4)])
%     plot([data_r(100:400,5),yp(100:400,5)])
%     title('Real (yellow) and predicted (purple) output')

%   Hit any key


%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
% %   
%        max_order = 60;
% %   
% %   As described in the help of subid we can determine "i" as follows:
% %   
% %       i = 2*(max_order)/(number of outputs)
% %       must be an integer
%         i = 2*(max_order)/l;
%

  nn= 20
    [A,du1,C,du2,K,R,AUX] = subid(data_r,[],i,nn,[],'SV',1);
    [A,du1,C,du2,K,R,AUX1] = subid_stable(data_r,[],i,nn,[],'SV',1);

  nn= 20  
    [Anp,du1,Cnp,du2,Knp,Rnp,AUX] = subid_stable(data_r(:,1:l/2),[],i,nn,[],'SV',1);
    [Awf,du1,Cwf,du2,Kwf,Rwf,AUX] = subid_stable(data_r(:,l/2+1:end),[],i,nn,[],'SV',1);
    
    %[A,du1,C,du2,K,R,AUX] = subid(data_r,[],i,nn,[],[],1);

  nn= 20  
    [Asp,du1sp,Csp,du2sp,Ksp,Rsp,AUXsp] = subid_sparse2(data_r,[],i,nn,[],'sv',1);

%     era = [];
%     for n = 1:20
%       [A,B,C,D,K,R] = subid(data_small,[],i,n,AUX,[],1);
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
% 
%     [A,du1,C,du2,K,R] = subid(data_small,[],i,nn,AUX,'sv');
% 
% %   The order is still clearly equal to 4.
% %   
% %   This concludes the stochastic demo.  You should now be ready
% %   to use subid on your own data.
% %   
% %   Note that other time series identification algorithms are:
% %   sto_stat, sto_alt and sto_pos.
% %   
% %   For a demonstration of combined identification, see sta_demo.

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
    x(:,i+1) = A*x(:,i) + K *( data_r(2000+(i-1),:)' - C*x(:,i) );
    y(:,i) = C*x(:,i);
    
    
    xsp(:,i+1) = Asp*xsp(:,i) + Ksp *( data_r(2000+(i-1),:)' - Csp*xsp(:,i) );
    ysp(:,i) = Csp*xsp(:,i);
end


% figure()
% for i=1:l
%     subplot(5,4,i)
%     plot(y(i,:));hold on;
%     plot(data_r(4000:5000,i))
% end

   
% figure()
% for i=1:n
%     subplot(5,4,i)
%     plot(x(i,:));hold on;
% end
x0 = x(:,end);

%% Forecast forward

T = 1000;
[n m] = size(A);
[n2 m2] = size(Awf)
x = zeros(n,T);
y = zeros(l,T);
x_wf = zeros(n2,T);
x_np = zeros(n2,T);
y_wf = zeros(l/2,T);
y_np = zeros(l/2,T);
xsp = zeros(n,T);
ysp = zeros(l,T);


% data_f = data_r(1000:2000,1:l)';
% data_fwf = data_r(1000:2000,l/2+1:end)';
% data_fnp = data_r(1000:2000,1:l/2)';


data_f = data_r(2000:3000,1:l)';
data_fwf = data_r(2000:3000,l/2+1:end)';
data_fnp = data_r(2000:3000,1:l/2)';
% x(:,1) = x0;
% x_wf(:,1) = x0;
% x_np(:,1) = x0;
for i = 1:T+1
    x(:,i+1) = A*x(:,i) + K *( data_f(:,i) - C*x(:,i) );
    y(:,i) = C*x(:,i);
    x_wf(:,i+1) = Awf*x_wf(:,i) + Kwf *( data_fwf(:,i) - Cwf*x_wf(:,i) );
    y_wf(:,i) = Cwf*x_wf(:,i);
    x_np(:,i+1) = Anp*x_np(:,i) + Knp *( data_fnp(:,i) - Cnp*x_np(:,i) );
    y_np(:,i) = Cnp*x_np(:,i);
    
    xsp(:,i+1) = Asp*xsp(:,i) + Ksp *( data_f(:,i) - Csp*xsp(:,i) );
    ysp(:,i) = Csp*xsp(:,i);
end

enp =  vecnorm(data_fwf - y_wf,2,1);
ewf =  vecnorm(data_fwf - y_wf,2,1);
e =  vecnorm(data_f - y,2,1);
%dd = eig(A);


figure()
for i=1:20
    subplot(5,4,i)
    plot(x(i,1:1000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:10
    subplot(5,4,i)
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(y(i,1:1000),'b');
    if i<=l/2
        plot(y_np(i,1:1000),'g');
    else
        plot(y_wf(i-l/2,1:1000),'r');
    end
    title(num2str(i))
end
sgtitle('Forecast outputs of 5 NP and 5 WF modes comparing to NP only and Wf only prediction')

% 
% figure()
% for i=1:l
%     plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
%     plot(y(i,1:1000),'g');
%     title(num2str(i))
% end

% writematrix(x,'NP_PCA_forecast_5_10.csv','Delimiter',',')
%
close all;

figure()
for i=1:10
    subplot(2,5,i)
    plot(x(i,1:500),'b','Linewidth',2);hold on;
    plot(xsp(i,1:500),'g','Linewidth',2);hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states NP + WF combined')


figure()
for i=1:10
    subplot(2,5,i)
    plot(data_f(i,1:500),'--k','Linewidth',1);hold on;
    
    if i<=l/2
        plot(y_np(i,1:500),'g','Linewidth',2);
    else
        plot(y_wf(i-l/2,1:500),'b','Linewidth',2);
    end
    plot(y(i,1:500),'r');
    plot(ysp(i,1:500),'m','Linewidth',2);
    title(num2str(i))
end
sgtitle('Forecast outputs of 5 NP and 5 WF modes comparing to NP only and Wf only prediction')


%%
% close all

ewf =  vecnorm(data_fwf(:,2:end) - y_wf(:,2:end),2,1);
enp =  vecnorm(data_fnp(:,2:end) - y_np(:,2:end),2,1);
eNP =  vecnorm(data_f(1:5,2:end) - y(1:5,2:end),2,1);
eWF =  vecnorm(data_f(6:10,2:end) - y(6:10,2:end),2,1);



figure()
plot(enp,'k');hold on
plot(eNP,'b');hold on
legend('NP only','combined')
title('NP error')

figure()
plot(ewf,'k');hold on
plot(eWF,'b');hold on
legend('WF only','combined')
title('WF error')

%%
Enp = sum(enp)
ENP = sum(eNP)
Ewf = sum(ewf)
EWF = sum(eWF)


%%


edges = 0;
eps = 0.5*1e-3;
for i=1:20
    for j=1:20
        if Asp(i,j)>eps
        edges = edges+ 1;
        end
        
    end
end
edges