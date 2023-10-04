clear all;
close all;
clc;

% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;

data1 = double(table2array(readtable('/home/mist/Documents/projects/Brain/code/tempn1.csv')));
data2 = double(table2array(readtable('/home/mist/Documents/projects/Brain/code/tempn2.csv')));
data3 = double(table2array(readtable('/home/mist/Documents/projects/Brain/code/tempn3.csv')));
data4 = double(table2array(readtable('/home/mist/Documents/projects/Brain/code/tempn4.csv')));
data5 = double(table2array(readtable('/home/mist/Documents/projects/Brain/code/tempn5.csv')));


disp('loading WF PCA projections')

[dims, time] = size(data1);
%%
n=10;
l=n;
% transpose for Umap
% data = data./vecnorm(data,2,2);
% data_r = data1(1:l,1:1000);
% data_small = data1(1:l,1:2000);
data_small = data1(1:l,1:1000);

% figure()
% plot(data_r(1,:));hold on
% plot(data_r(2,:));
% plot(data_r(3,:));
% plot(data_r(4,:));
% % plot(data_r(:,5));
% title('Outputs 1-5');
    


%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
       max_order = 50;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       must be an integer
        p = 2*(max_order)/l;

  nn= 10;
  AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

 [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');

%% Reconstruction
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
    
    % x(:,i+1) = A*x(:,i) + 0*K *( data_small(:,i) - C*x(:,i) );
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
end
% x0 = x(:,end);

%% Forecast forward
close all;
T = 5000;
data_f = data1(1:l,2000:end);

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
    subplot(2,5,i)
    
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
legend('unstable','stable')


%%
figure()
for i=1:n
    subplot(5,4,i)
    plot(xs(i,4500:5000));hold on;
    title(num2str(i))
end
sgtitle('Forecast latent states')
%%

% st = 1000;
% data_small = data1(1:l,st:st+499);
% 
% st = 1500;
% data_small = data1(1:l,st:st+499);
% 
%  [A,du1,C,du2,K,R,AUX] = subid(data_small,[],i,nn,[],[],1);
% 
%  [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],i,nn,AUX,'sv');
% 
% dd2 = eig(A)
% ds2 = eig(As)
% figure()
% plot(dd,'ob');hold on
% plot(ds,'or');
% plot(dd2,'*b');hold on
% plot(ds2,'*r');
% legend('unstable','stable')
%%

nn=6
Ds=[];
si = 500;
ss = 10000/si; 
for j = 1:ss
    
data_small = data1(1:l,si*(j-1)+1 : j*si);
AUX=[];
p
 % [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');

    %dd2 = eig(A)
    % Ds = [Ds,eig(A)];
    Ds = [Ds,eig(As)];
eig(A)
end
%%
Ds2=[];
for j = 1:ss
    
data_small = data2(1:l,si*(j-1)+1 : j*si);
AUX=[];
p
 % [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');

    %dd2 = eig(A)
    % Ds2 = [Ds2,eig(A)];
    Ds2 = [Ds2,eig(As)];

end
%%
Ds3=[];
for j = 1:ss
    
data_small = data3(1:l,si*(j-1)+1 : j*si);
AUX=[];
p
 % [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');

    %dd2 = eig(A)
    % Ds3 = [Ds3,eig(A)];
    Ds3 = [Ds3,eig(As)];

end

%%

figure()
plot(Ds,'or'); hold on;
plot(Ds2,'ob');
plot(Ds3,'og');

%%

%%
close all;

cDs = Ds/dt;
cDs2 = Ds2/dt;
cDs3 = Ds3/dt;
t= 1:1:length(Ds);
figure()
plot(t,abs(imag(cDs)),'or'); hold on;
plot(t,abs(imag(cDs2)),'ob');hold on;
plot(t,abs(imag(cDs3)),'og');hold on;
