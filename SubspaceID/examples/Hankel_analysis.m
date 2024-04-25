%% Requirement: i > n, other than that no notable effect for fized window length 

clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;

%% read data from server
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


%% S
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
       max_order = 100;
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


t0 = 20000;
T  = 500;
Ds = [];
D = [];
nn = 9;
l = 10;

for i = 1:5
    i
    p = 10 + 2*i
    % data = data1(1:l,Ts*(i-1)+1 : i*Ts);
    data = yn(1:l,t0+1: t0+T);

    AUX=[];
    [A,du1,C,du2,K,R,AUX] = subid(data,[],p,nn,[],[],1);

    [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn,AUX,'sv');

    %As = As - Ks*Cs;

    [E,V] = eig(As);
    A = logm(A)/dt;
    As = logm(As)/dt;

    D = [D,eig(A)];
    Ds = [Ds,eig(As)];

end

%

figure()
plot(abs(imag(Ds))','or'); hold on;
title('Stable continuous time eigenvalues')

figure()
plot(abs(imag(D))','ob'); hold on;
title('unstable continuous')

%% Embeddings :: 2i >= 40