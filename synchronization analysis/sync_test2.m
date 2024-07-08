clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;


path = "/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/AB_0032/2024-03-14/1";
upath = '/violet/svdSpatialComponents.npy';
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
close all;

i = 10;
T = 1000;
t0 = 20000;

Sn = yn(i,t0:t0+T);
figure()
pspectrum(Sn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 15]);
%%
figure()
fsst(Sn,fs,'yaxis')

%%
i=250;

Sn = Vv(t0:t0+T,i);
figure()
pspectrum(Sn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 15]);

figure()
fsst(Sn,fs,'yaxis')
%% bunch up
nn=50;
C = ones(1,nn);
CSn = C*Vv(t0:t0+T,1:nn)';
figure()
pspectrum(Sn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 15]);

figure()
fsst(Sn,fs,'yaxis')



%%  pixel specific
Uu = reshape(Uu(:,:,1:500),[560*560,500]);
% time
i=10000;
Im =  reshape(Uu*Vv1(i,:)',560,560);

% filtered and rank 10
Im2 =  reshape(Uu(:,1:5)*y(i,1:5)',560,560);


Im3 =  reshape(Uu(:,1:5)*yn(1:5,i),560,560);


%
P=[250,400];
close all;
figure()
image(Im');hold on;
plot(P(1),P(2),'ro')
colorbar
%%

figure()
image(Im2');hold on;
plot(P(1),P(2),'ro')
colorbar

figure()
image(1e2*Im3');


J = imadjust(Im3);

figure()
imshow(J);hold on;
plot(P(1),P(2),'ro')
colorbar
%%
F=0;
Ff=0;
Ffn=0;

t0=10000;
T=500;
for t = t0:t0+T

Im =  reshape(Uu(:,1:100)*Vv1(t,1:100)',560,560);    

% filtered low rank
Imf =  reshape(Uu(:,1:5)*y(t,1:5)',560,560);

Imfn =  imadjust(reshape(Uu(:,1:5)*yn(1:5,t),560,560));


f=0;

ff=0;
ffn=0;

for i = -3:1:3
    for j = -3:1:3
        f = f + Im(P(1)+i,P(2)+j)/50;
        ff = ff + Imf(P(1)+i,P(2)+j)/50;
        ffn = ffn + Imfn(P(1)+i,P(2)+j)/50;


    end
end
F=[F,f];
Ff=[Ff,ff];
Ffn=[Ffn,ffn];
end
%%
%

figure()
plot(F)

% Spectrum on pixel
figure()
pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(F,fs,'yaxis')


%%

figure()
plot(Ff)

% Spectrum on pixel
figure()
pspectrum(Ff,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(Ff,fs,'yaxis')
%%
close all;
figure()
plot(Ffn)

% Spectrum on pixel
figure()
pspectrum(Ffn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(Ffn,fs,'yaxis')

%%
close all;

figure()
plot(F)

% Spectrum on pixel
figure()
% pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
%     'MinThreshold',-50,'FrequencyLimits',[0, 10]);
pspectrum(F,fs,'spectrogram','Leakage',0.1,'OverlapPercent',99, ...
    'MinThreshold',-100,'FrequencyLimits',[0, 10]);


figure()
fsst(F,fs,'yaxis')