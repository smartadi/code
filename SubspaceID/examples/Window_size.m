%% Window size
%% Need to randomize over the entire dataset, location of sampling effects frequencies more than window size with high embedding size

clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1";
upath = '/corr/svdSpatialComponents_ortho.npy';
upath = append(path,upath);
vpath = '/corr/svdTemporalComponents_ortho.npy';
vpath = append(path,vpath);


Uu = readUfromNPY(upath);
Vv1 = readVfromNPY(vpath);

[TT,dims]= size(Vv1);

Vv = Vv1./vecnorm(Vv1,2,2);

disp('loading WF PCA projections')

data1=Vv';
N=10000;

l=10;
ll=100;
S = Vv1(1:50000,1:ll);

%% filter
close all;
fc = 10;
n = 8;
fs = 35;
[b,a] = butter(6,fc/(fs/2));
y = filter(b,a,S);


yn = (y./vecnorm(y,2,2))';

%% sample
n = 5;
l = n;
Ts = 1000;
data_small = yn(2:l,1:Ts);
nn = 8

%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
       max_order = 20;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       must be an integer
       p = 2*(max_order)/l;

  % nn= 10;

  AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

 [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');


%%
dd = eig(A)
ds = eig(As)
figure()
plot(dd,'ob');hold on
plot(ds,'or');
legend('unstable','stable')


%% Icremental window
LL = 1500;
Ds=[];
cDs=[];
% shift by sT data points
sT = 1;

t0 = 35000
Ts = 500;
nn = 9;
l = 10
p = 20
for i = 1:LL
    % data = data1(1:l,t0-Ts/2 +  sT*(i-1)+1 : t0-Ts/2+ sT*(i-1)+Ts);
    i;
    data = yn(1:l,t0 : t0+Ts+i);
    AUX=[];
    [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn,AUX,'sv',1);
    %[As,du1s,Cs,du2s,Ks,Rs] = subid(data,[],p,nn,AUX,'sv',1);
    A = logm(As)/dt;
    cDs = [cDs,eig(A)];

    Ds = [Ds,eig(As)];
end

%
% close all;
% figure()
% plot(Ds,'or'); hold on;
% axis([-1 1 -1 1])
% title('Discrete time stable eigenvalues')
% 
% 
% cDs = (Ds-1)/dt;
t= Ts+1:1:Ts+LL;
% 
% figure()
% plot(t,abs(imag(cDs)),'or'); hold on;
% title('Stable continuous time frequencies')
% 
% 
% 
% figure()
% plot(cDs,'or'); hold on;
% title('Stable continuous time eigenvalues')


% %%
% CDs = log(Ds)/dt;
% 
% close all;
% figure()
% plot(t,abs(imag(CDs)),'or'); hold on;
% title('Stable discrete time frequency')
%
close all;
% figure()
% plot(t,abs(imag(Ds)),'or'); hold on;
% title('Stable DT frequencies')

figure()
plot(t,abs(imag(cDs)),'or'); hold on;
title('frequencies for increasing window size')

%%
% close all;
% figure()
% plot(abs(imag(Ds))'); hold on;
% title('Stable DT frequencies')
% %%
% AAA = [zeros(8,Ts),abs(imag(Ds))];
% writematrix(AAA,'freq_sliding.csv')
