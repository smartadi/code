clc;
close all;
clear all;
githubDir = "/home/nimbus/Documents/Brain/"

% Script to analyze widefield/behavioral data from 

addpath(genpath(fullfile(githubDir, 'widefield'))) % cortex-lab/widefield
addpath(genpath(fullfile(githubDir, 'Pipelines'))) % SteinmetzLab/Pipelines
addpath(genpath(fullfile(githubDir, 'npy-matlab'))) % kwikteam/npy-matlab

%%
%70
mn = 'AB_0032'; td = '2024-07-26'; 
en = 1;

%73
% mn = 'AL_0034'; td = '2024-07-29'; 
% en = 1;
% 
% % 69
% mn = 'AL_0033'; td = '2024-07-25'; 
% en = 1;

serverRoot = expPath(mn, td, en)

%% preprocess video

% colors = {'blue', 'violet'};
% computeWidefieldTimestamps(serverRoot, colors);



%% svdViewer
% load(fullfile(serverRoot, 'blue', 'dataSummary.mat'));
% svdViewer(U, Sv(1:nSV), V, 1/mean(diff(t))); 

%% process hemodynamic correction
% nSV = 500;
% 
% [U, V, t, mimg] = hemoCorrect(serverRoot, nSV);

%% load directly (if you had already run correction)
nSV = 500;

[U, V, t, mimg] = loadUVt(serverRoot, nSV);

%% correlation map
pixelCorrelationViewerSVD(U,V)

%% movie
sigName = 'lightCommand';
[tt, v] = getTLanalog(mn, td, en, sigName);

tInd = 1;
traces(tInd).t = tt;
traces(tInd).v = v;
traces(tInd).name = sigName;
traces(tInd).lims = [0 5];

% tInd = 2;
% traces(tInd).t = tt;
% traces(tInd).v = vel;
% traces(tInd).name = 'wheelVelocity';
% traces(tInd).lims = [-3000 3000];

movieWithTracesSVD(U, V, t, traces, [], []);

%%

% matchBlocks2Timeline(mn,td,[3 4],[])


%% 

% dV = [zeros(size(V,1),1) diff(V,[],2)];

%%

% [newU, newV] = dffFromSVD(U, V, mimg);


%%
stimTimes = tt(v(2:end)>2 & v(1:end-1)<=2);
ds = find(diff([0;stimTimes])>0.05);
stimStarts = stimTimes(ds);
stimEnds = stimTimes(ds(2:end)-1); 
%%
stimDur = stimEnds-stimStarts(1:end-1);

%%

% pixelTuningCurveViewerSVD(U, V(:,1:end-1), t, stimStarts(1:end-1), stimDur, [-1 3])



%%

h  = histogram(stimDur,'BinWidth',0.05);
%%
[N,edges,bin] = histcounts(stimDur,20)


events = edges(bin) + 0.1;

%%
% pixelTuningCurveViewerSVD(U, V(:,1:end-1), t, stimStarts(1:end-1),events, [-1 3])
pixelTuningCurveViewerSVD(U, V, t, stimStarts(1:end-1),events, [-1 5])

%% sort
uu = reshape(U(:,:,1:500),560*560,500);


pixel = [230,220;
        360,130;
        170,330;
        230,80]
%%
% nFrames = length(V)
% F=[];
% F2=[];
% for i=1:nFrames
%     Im = reshape(uu*V(:,i),[560,560])+mimg;
%     Im2 = reshape(uu*V(:,i),[560,560]);
%     G=[];
%     G2=[];
%     for j = 1 :length(pixel)
%         G = [G;Im(pixel(j,1),pixel(j,2))];
%         G2 = [G2;Im2(pixel(j,1),pixel(j,2))];
%     end
%     i
%     F = [F,G];
%     F2 = [F2,G2];
% end
% %%
% save('input_pixel_vars.mat','F','F2');
load('../input_pixel_vars.mat');


%% 

figure()
plot(F2')


%%
% Lf = readNPY('Lf.npy')';
% Lp = readNPY('Lp.npy')';
% 
% Lff = readNPY('Lff.npy');
% Lpf = readNPY('Lpf.npy');
% 
% 
% 
% Lf = [zeros(4,2*35),Lf];
% Lp = [zeros(4,2*35),Lp];
% 
% Lf(:,1)=[];
% Lp(:,1)=[];
% F2(:,1)=[];
% Lff(:,1)=[];
% Lpf(:,1)=[];
% %%
% close all
% 
% 
% t0=500;
% T = 10000;
% 
% figure()
% subplot(3,1,1)
% plot(t(T:T+t0),F2(1,T:T+t0));
% xline(stimStarts)
% xlim([t(T),t(T+t0)])
% ylabel('mean pixel values')
% 
% subplot(3,1,2)
% plot(t(T:T+t0),Lf(:,T:T+t0));
% xline(stimStarts)
% xlim([t(T),t(T+t0)])
% ylabel('freq power ratio for 5 pixels')
% 
% 
% subplot(3,1,3)
% plot(t(T:T+t0),Lff(:,T:T+t0));
% xline(stimStarts)
% xlim([t(T),t(T+t0)])
% ylabel('freq power ratio over window lengths')
% legend('1s','2s','3s','4s','5s')
% 
% %% Pascha experiments 
% Fs = 1000;            % Sampling frequency                    
% T = 1/Fs;             % Sampling period       
% L = 1500;             % Length of signal
% t = (0:L-1)*T;
% S = 0.8 + 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
% X = S + 2*randn(size(t));
% Y = fft(S);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs/L*(0:(L/2));
% plot(f,P1,"LineWidth",3) 
% title("Single-Sided Amplitude Spectrum of S(t)")
% xlabel("f (Hz)")
% ylabel("|P1(f)|")
% 
% 
% M=50
% figure()
% pspectrum(X,Fs,"spectrogram", ...
%     TimeResolution=M/Fs,OverlapPercent=0.1, ...
%     Leakage=0.9)
% title("pspectrum")
% cc = clim;
% xl = xlim;
% close all

%% signal
Uu=reshape(U,560*560,500);
%%
p = [100,100];
j=1
% for i=1:N
    Im = Uu((p(j,2)-1)*560 + p(j,1),1:50)*V(1:50,:);
% end

%%
close all
figure()
plot(t,Im)
%%

