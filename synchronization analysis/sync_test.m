clear all;
close all;
clc;
% dt = 1/35;
dt = 0.0285;
t = 0:dt:dt*1000;


path = "/run/user/1001/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1";
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
close all;

i=10;
T=1000;
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
% time
i=10000;
Im =  reshape(Uu*Vv1(i,:)',560,560);

% filtered and rank 10
Im2 =  reshape(Uu(:,1:5)*y(i,1:5)',560,560);


Im3 =  reshape(Uu(:,1:5)*yn(1:5,i),560,560);


%
P=[225,425];
close all;
figure()
image(Im');hold on;
plot(P(1),P(2),'ro')
colorbar


for i = -10:1:10
    for j = -10:1:10
        Im(P(1)+i,P(2)+j)=0;
    end
end
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
Fn=0;

t0=10000;
T=5000;
for t = t0:t0+T

Im =  reshape(Uu(:,1:100)*Vv1(t,1:100)',560,560);    


Imfn =  imadjust(reshape(Uu(:,1:5)*yn(1:5,t),560,560));


f=0;

fn=0;

for i = -10:1:10
    for j = -10:1:10
        f = f + Im(P(1)+i,P(2)+j);
        fn = fn + Imfn(P(1)+i,P(2)+j);


    end
end
F=[F,f];
Fn=[Fn,fn];
end
%%
%

figure()
plot(F);hold on

% Spectrum on pixel
figure()
pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(F,fs,'yaxis')


%

figure()
plot(Fn)

% Spectrum on pixel
figure()
pspectrum(Fn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(Fn,fs,'yaxis')
% %%
% close all;
% figure()
% plot(Ffn)
% 
% % Spectrum on pixel
% figure()
% pspectrum(Ffn,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
%     'MinThreshold',-50,'FrequencyLimits',[0, 10]);
% 
% figure()
% fsst(Ffn,fs,'yaxis')
% 
% %%
% close all;
% 
% figure()
% plot(F)
% 
% % Spectrum on pixel
% figure()
% % pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
% %     'MinThreshold',-50,'FrequencyLimits',[0, 10]);
% pspectrum(F,fs,'spectrogram','Leakage',0.1,'OverlapPercent',99, ...
%     'MinThreshold',-100,'FrequencyLimits',[0, 10]);
% 
% 
% figure()
% fsst(F,fs,'yaxis')
%%
% Spectrum on pixel
close all;
figure()
set(gca,'ColorScale','log');
pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[3, 6]);




figure()
fsst(F,fs,'yaxis')


figure()
plot(Fn)

figure()
plot(F)
%%
F=F/1e6;
%%
close all;
clc;
[sp,fp,tp] = pspectrum(F,fs,"spectrogram",'FrequencyLimits',[0, 10]);


waterfall(fp,tp,sp')
set(gca,XDir="reverse",Zscale='log',View=[60 60])
ylabel("Time (s)")
xlabel("Frequency (Hz)")

%%
close all;
figure()
set(gca,'ColorScale','log');
pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[0, 10]);

figure()
fsst(F,fs,'yaxis')


figure()
plot(F)



F_short = F(1:500);
close all;
figure()
set(gca,'ColorScale','log');
pspectrum(F_short,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-50,'FrequencyLimits',[3, 10]);

figure()
fsst(F_short,fs,'yaxis')


figure()
plot(F_short)


%%
close all;
figure()
plot(Vv1(t0:t0+T,1:10));




figure()
plot(Vv1(t0:t0+500,1:10));
figure()
plot(Vv(t0:t0+500,1:10));


figure()
plot(sum(Vv(t0:t0+500,1:10),2))

figure()
plot(F(1:500));
%%
F_short = F(1:500);
%%
figure()
pspectrum(F,fs,'spectrogram','Leakage',1,'OverlapPercent',99, ...
    'MinThreshold',-100,'FrequencyLimits',[1, 10]);


figure()
fsst(F_short,fs,'yaxis')
