%
% Demo file for Stochastic Subspace identification
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%

clc
echo off
disp(' ')
disp(' ')
disp('                SUBSPACE IDENTIFICATION ')
disp('               -------------------------')
disp(' ')

disp('   This file will guide you through the world of time series')
disp('   modeling with subspace identification algorithms.')
disp('  ')
disp('  ')
disp('   Hit any key to continue')

pause

clc
echo on
%
%   Consider a multivariable fourth order system a,k,c,r
%   which is driven by white noise and generates an output y: 
%   
%               x_{k+1} = A x_k + K e_k
%                y_k    = C x_k + e_k
%              cov(e_k) = R
echo off
disp('loading WF PCA projections')
%data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_UMAP_projections_small.csv')))';
%data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_KPCA_projections_small.csv')));
% data = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_PCA_projections_small.csv')));

data = real(yWF(:,1:5000)');

disp('   Hit any key to continue')

[dims, time] = size(data);
pause
echo off

%    a = [ 0.999505  0.00984985;-0.09849847  0.96995546];
%    k = [0.1;0.1];
%    c = [1,0];
%    r = [0.01];
%    l = 1; 				% Number of outputs

l=5
% transpose for Umap

    


%   Hit any key
pause
clc

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
pause
clc
%
%   The subspace algorithms is now e asily started.
%   Note the dummy outputs (du1 and du2) where normally the B and D 
%   matrices are located.  Also note that u = [];
%   When prompted for the system order you should enter 4.
%   
    [A,du1,C,du2,K,R] = subid(data,[],i);

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
    [yp,erp] = predic(data,[],A,[],C,[],K);
%
%   erp contains the error per output in percentage:    
    erp    
%   While yp contains the predicted output: 
    figure()
    plot([data(100:400,1),yp(100:400,1)]);hold on
    plot([data(100:400,2),yp(100:400,2)])
    plot([data(100:400,3),yp(100:400,3)])
    plot([data(100:400,4),yp(100:400,4)])
    plot([data(100:400,5),yp(100:400,5)])
    title('Real (yellow) and predicted (purple) output')

%   They coincide well.    
%   Hit any key
pause
clc

%   In many practical examples, the gap in the singular value plot 
%   is not as clear as in this example.  The order decision then becomes
%   less trivial.  There is however an nice feature of sto_pos which allows
%   for fast computation of systems with different orders. 
%   This can be done through the extra variable AUX which appears
%   as an input as well as an output argument of subid.
%   The last parameter (1) indicates that the algorithm should run silently.
  
    [A,du1,C,du2,K,R,AUX] = subid(data,[],i,2,[],[],1);
    era = [];
    for n = 1:5
      [A,B,C,D,K,R] = subid(data,[],i,n,AUX,[],1);
      [yp,erp] = predic(data,[],A,[],C,[],K);
      era(n,:) = erp;
    end
    
%   Hit any key
pause
clc
%           
%   We have now determined the prediction errors for all systems 
%   from order 1 through 6.
%   Plotting these often gives a clearer indication of the order:
%   
    subplot;
    bar([1:5],era);title('Prediction error');
    xlabel('System order');
    
%   It now even becomes more clear that the system order is 4.
%    


%   A last feature we would like to illustrate is that subid (or
%   the other stochastic identification algorithms) also
%   can work with another basis.     
%   The order of the system is then equal to the number of singular
%   values different from zero (instead of the number of angles
%   different from 90 degrees):

    [A,du1,C,du2,K,R] = subid(data,[],i,[],AUX,'sv');

%   The order is still clearly equal to 4.
%   
%   This concludes the stochastic demo.  You should now be ready
%   to use subid on your own data.
%   
%   Note that other time series identification algorithms are:
%   sto_stat, sto_alt and sto_pos.
%   
%   For a demonstration of combined identification, see sta_demo.
echo off

%%
% predict future
%            xp_{k+1} = A xp_k + B u_k + K (yp_k - C xp_k - D u_k)
%              yp_k   = C xp_k + D u_k
clc;
close all
T = 100;
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);


for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data(100+(i-1),:)' - C*x(:,i) );
    y(:,i) = C*x(:,i);
end


figure()
for i=1:l
    subplot(5,1,i)
    plot(y(i,:));hold on;
    plot(data(400:500,i))
end


figure()
for i=1:n
    subplot(5,1,i)
    plot(x(i,:));hold on;
end
x0 = x(:,end);

% Forecast forward

T = 5000;
[n m] = size(A);
x = zeros(n,T);
y = zeros(l,T);
% data_f = data(1:l,5000:10000);
data_f =real(yWF(:,5000:end));
x(:,1) = x0;

for i = 1:T
    x(:,i+1) = A*x(:,i) + K *( data_f(:,i) - C*x(:,i) );
    y(:,i) = C*x(:,i);
end



dd = eig(A);


figure()
for i=1:n
    subplot(5,1,i)
    plot(x(i,1:1000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:l
    subplot(5,1,i)
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(y(i,1:1000),'b');
    title(num2str(i))
end
sgtitle('Forecast outputs GT modes')

figure()
for i=1:l
    plot(data_f(i,1:1000),'k','Linewidth',2);hold on;
    plot(y(i,1:1000),'g');
    title(num2str(i))
end

%%
close all;
figure()
for i=1:n
    subplot(5,1,i)
    plot(x(i,4500:5000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')


figure()
for i=1:l
    subplot(5,1,i)
    plot(data_f(i,4500:5000),'k','Linewidth',2);hold on;
    plot(y(i,4500:5000),'b');
    title(num2str(i))
end
sgtitle('Forecast outputs UMAP modes')