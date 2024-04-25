% 
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996
%           Generation of Figure 3.10 on Page 83
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%
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
S = Vv1(1:50000,1:l);

% filter

fc = 10;
n = 8;
fs = 35;
[b,a] = butter(6,fc/(fs/2));
y = filter(b,a,S);


y = (y./vecnorm(y,2,2))';

%%
Nmax=5000; 				% Maximum j
Nstep=10; 				% j step
Nstart=100; 				% Start j
i=20; 					% #block rows

% The system
% a = [0.6,0.6,0;-0.6,0.6,0;0,0,0.4];
% k = [0.1706;-0.1507;0.2772];
% c = [0.7831,0.5351,0.9701];
% r = 6.3663;

% Generate the data
%en=randn(Nmax+2*i,1)*sqrt(r); 		% The noise
%y=dlsim(a,k,c,1,en); 			% The output

ax=[Nstart:Nstep:Nmax]; 		% The axis
ne=max(size(ax)); 			% Number of points
R=[]; 					% R factor
s1=[];s2=[];s3=[];

nn=0;
Nb=0;
%%
for k=1:ne
  disp(['   Computing for j = ',num2str(ax(k)),' points'])
  % To save time, we compute R recursively
  if k == 1
    Ye = blkhank(y(1:Nmax),2*i,ax(1));
    Nn = ax(1);
  else
    Ye = blkhank(y(ax(k-1)+1:Nmax+2*i),2*i,ax(k)-ax(k-1));
    Nn=ax(k);
  end
  R=[R*sqrt(Nb),Ye]/sqrt(Nn);
  R=triu(qr(R'))';
  R=R(1:2*i,1:2*i);
  
  % Split R
  R11=R(1:i,1:i);
  R21=R(i+1:2*i,1:i);
  R22=R(i+1:2*i,i+1:2*i);

  % The projection :
  Ob=R21;
		
  % PC algorithm :
  mm1=R21*R11';
  s1(:,k)=svd(mm1);

  % UPC algorithm :
  mm2=Ob;
  s2(:,k)=svd(mm2);

  % CVA algorithm :
  W1=inv(sqrtm(R21*R21'+R22*R22'));
  mm3=W1*Ob;
  s3(:,k)=svd(mm3);

  Nb=Nn;
end
%%
close all;
% Plot everything
hold off
subplot
subplot(311)
plot(ax,s1','-')
axis([0,Nmax,0,1]);
title('Singular Values PC algorithm') 
subplot(312)
plot(ax,s2','-')
axis([0,Nmax,0,1]);
title('Singular Values UPC algorithm') 
subplot(313)
plot(ax,acos(s3')/pi*180,'-')
axis([0,Nmax,0,90]);
title('Principal Angles CVA algorithm') 
xlabel('Number of block rows j')
ylabel('degrees')









