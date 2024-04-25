%% Effect of laltent space dimension
% nor l=10 take n=9

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


%%
nn = 6

% max_order = 25;
% p = 2*(max_order)/l;

p=25

Ts = 1000;
data_small = yn(1:l,1:Ts);

AUX=[];

[A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');


%% Icremental dimensions
LL = 900;
Ds=[];

% shift by sT data points
sT = 1;
t0 = 4000
Ts = 1000;
nn = 4;
d = 30;

Ds = zeros(d+nn,d);
cDss = zeros(d+nn,d);


data = yn(1:l,t0 : t0+Ts);
for i = 1:d
    % data = data1(1:l,t0-Ts/2 +  sT*(i-1)+1 : t0-Ts/2+ sT*(i-1)+Ts);
    i

    % data = data1(1:l,t0 : t0+Ts);
    AUX=[];
    % [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn+i-1,AUX,'sv',1);
    [As,du1s,Cs,du2s,Ks,Rs] = subid(data,[],p,nn+i-1,AUX,'sv',1);

    Ds(1:nn+i-1,i) = eig(As);

     A = logm(As)/dt;

     cDss(1:nn+i-1,i) = eig(A);
    
end

t= 1:1:d;

%
close all;
figure()
plot(t,abs(imag(Ds)),'*r'); hold on;
title('Stable DT frequencies')
    
figure()
plot(t,abs(imag(cDss)),'or'); hold on;
xlabel('system dimension')
ylabel('frequency')
title('Continuous time frequency')
%%

close all;
Ts = 1000;
data = yn(1:l,t0 : t0+Ts);


% [A,du1,C,du2,K,R] = subid(data,[],p)
% pause


    [A,du1,C,du2,K,R,AUX] = subid(data,[],p,2,[],[],1);
    era = [];
    for n = 1:d
      [A,B,C,D,K,R] = subid(data,[],p,n,AUX,[],1);
      [yp,erp] = predic(data,[],A,[],C,[],K);
      era(n,:) = erp;
    end
    %
    subplot;
    bar([1:d],era);title('Prediction error');
    xlabel('System order');


    %%
        [A,du1,C,du2,K,R] = subid(data,[],p,[],AUX,'sv');

%%  
Ts = 500;
t0 = 30000;

l = 10
p = 20


data = yn(2:l,t0 : t0+Ts);
d=20
% [ersa,erpa] = allord(data,[],p,[1:d],AUX);
[ersa,erpa] = allord(data,[],p,[1:d],[]);


%%

% close all;
% figure()
% plot(eig(A),'or'); hold on;
% axis([-1 1 -1 1])
% title('Discrete time stable eigenvalues')

%%

Ts = 500;
t0 = 500;
nn = 7;
data = yn(2:10,t0 : t0+Ts);
[A,du1,C,du2,K,R] = subid(data,[],p,nn,AUX,'sv');
vecnorm(eig(A),2,2)



%%

% close all;
% T = 1000;
% 
% [n m] = size(A);
% x = zeros(n,T);
% y = zeros(l,T);
% 
% xs = zeros(n,T);
% ys = zeros(l,T);
% data_f = yn(1:l,1000:2000);
% tt = 1000:dt:1000+dt*1000;
% for i = 1:T
%     x(:,i+1) = A*x(:,i) + K *( data_f(:,i) - C*x(:,i) );
%     yy(:,i) = C*x(:,i);
% end
% 
% 
% figure()
% plot(tt,x)
% 
% figure()
% plot(tt(1:end-1),yy);hold on
% plot(tt,data_f,'Linewidth',2)
% %% Pred
% T = 1000;
% 
% [n m] = size(A);
% xp = zeros(n,T);
% yp = zeros(l,T);
% xp(:,1) = x(:,end);
% data_f = yn(1:l,2000:3000);
% tt = 1000:dt:1000+dt*1000;
% for i = 1:T
%     xp(:,i+1) = A*xp(:,i);% + K *( data_f(:,i) - C*x(:,i) );
%     yp(:,i) = C*xp(:,i);
% end
% 
% 
% figure()
% plot(tt,xp)
% 
% figure()
% plot(tt(1:end-1),yp);hold on
% plot(tt,data_f,'Linewidth',2)