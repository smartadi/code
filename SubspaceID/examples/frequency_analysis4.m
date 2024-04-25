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

mpath = append(path,'/face_proc.npy');

%Mm = readUfromNPY(mpath)

Uu = readUfromNPY(upath);
Vv1 = readVfromNPY(vpath);

[TT,dims]= size(Vv1);

Vv = Vv1./vecnorm(Vv1,2,2);

disp('loading WF PCA projections')

data1=Vv';
N=10000;

l=10;
S = Vv1(1:10000,1:l);

%% filter
close all;
fc = 10;
n = 8;
fs = 35;
[b,a] = butter(6,fc/(fs/2));
y = filter(b,a,S);


% figure()
% semilogy(tt,YY(L/2+1:end,:),"LineWidth",3)
% title("fft Spectrum in the Positive and Negative Frequencies")
% xlabel("f (Hz)")
% ylabel("|fft(X)|")

yn = (y./vecnorm(y,2,2))';

% figure()
% plot(y(1:1000,1:10));hold on;
% 
% %
% figure()
% plot(yn(1:1000,2));hold on;
% plot(Vv(1:1000,2))

%
% Y = fft(y);
% 
% YY = abs(fftshift(Y));
% tt= Fs/L*(0:L/2-1);

% figure()
% plot(tt,a(L/2+1:end,:),"LineWidth",3)
% title("fft Spectrum in the Positive and Negative Frequencies")
% xlabel("f (Hz)")
% ylabel("|fft(X)|")
% 

% figure()
% semilogy(tt,YY(L/2+1:end,:),"LineWidth",3)
% title("fft Spectrum in the Positive and Negative Frequencies")
% xlabel("f (Hz)")
% ylabel("|fft(X)|")
%%
nn = 6

max_order = 25;
p = 2*(max_order)/l;

Ts=1000;
data_small = yn(1:l,1:Ts);

  AUX=[];

 [A,du1,C,du2,K,R,AUX] = subid(data_small,[],p,nn,[],[],1);

 [As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data_small,[],p,nn,AUX,'sv');


%%
LL = 3000
Ds=[];
cDss=[];
% shift by sT data points
sT = 1;

t0 = 6000
Ts = 1000;
nn = 6;
for i = 1:LL
    data = yn(1:l,t0-Ts/2 +  sT*(i-1)+1 : t0+Ts/2+ sT*(i-1));
    AUX=[];
    %[As,du1s,Cs,du2s,Ks,Rs] = subid_stable(data,[],p,nn,AUX,'sv',1);
    [As,du1s,Cs,du2s,Ks,Rs] = subid(data,[],p,nn,AUX,'sv',1);

    A = logm(As)/dt;
    cDss = [cDss,eig(A)];

    Ds = [Ds,eig(As)];
end
%%
close all;
figure()
plot(Ds,'or'); hold on;
axis([-1 1 -1 1])
title('Discrete time stable eigenvalues')

cDs = log(Ds)/dt;
t= 1:1:LL;

figure()
plot(t,abs(imag(cDs)),'or'); hold on;
title('Stable continuous time frequencies')

figure()
plot(cDs,'or'); hold on;
title('Stable continuous time eigenvalues')


%%
% close all;
figure()
plot(t,abs(imag(Ds)),'or'); hold on;
title('Stable DT frequencies')

%%
close all;
figure()
plot(t,abs(imag(Ds))); hold on;
title('Stable DT frequencies')


figure()
plot(t,abs(imag(cDss))); hold on;
title('Stable continuous time frequencies')


%%
close all

figure()
subplot(6,1,1)
plot(t,abs(imag(cDss))); hold on;
title('Stable continuous time frequencies')
subplot(6,1,2)
plot(t,yn(1,t0:t0+LL-1)); hold on;
subplot(6,1,3)
plot(t,yn(2,t0:t0+LL-1)); hold on;
subplot(6,1,4)
plot(t,yn(3,t0:t0+LL-1)); hold on;
subplot(6,1,5)
plot(t,yn(4,t0:t0+LL-1)); hold on;
subplot(6,1,6)
plot(t,yn(5,t0:t0+LL-1)); hold on;

