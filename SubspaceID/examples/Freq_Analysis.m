%% Frequency analysis on the raw data 

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

%% power spectrum
% close all;
 

L=1000;
S = Vv1(1:L,1:100);

Fs = 35;

f = Fs/L*(0:(L-1));

figure()
plot(f,S,"LineWidth",3) 
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")
%%

Y = fft(S);
P2 = abs(Y/L);
P1 = P2(1:L-1);
P1(2:end-1) = 2*P1(2:end-1);

figure()
plot(f,Y,"LineWidth",3) 
title("Single-Sided Amplitude Spectrum of S(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")


figure()
plot(Fs/L*(-L/2:L/2-1),abs(fftshift(Y)),"LineWidth",3)
title("fft Spectrum in the Positive and Negative Frequencies")
xlabel("f (Hz)")
ylabel("|fft(X)|")


power = abs(Y).^2/L;    % power of the DFT

figure()
plot(f,power)
xlabel('Frequency')
ylabel('Power')
%%

[pxx,f] = pspectrum(abs(Y),35);

figure()
plot(f,pxx)
xlabel('Frequency (Hz)')
ylabel('Power Spectrum (dB)')
title('Default Frequency Resolution')

%%

Fs = 35;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
t = (0:L-1)*T;        % Time vector


Y = fft(S);

figure()
plot(Fs/L*(0:L-1),abs(Y),"LineWidth",3)
title("Complex Magnitude of fft Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")

figure()
plot(Fs/L*(-L/2:L/2-1),abs(fftshift(Y)),"LineWidth",3)
title("fft Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")

a = abs(fftshift(Y));
tt= Fs/L*(0:L/2-1);
figure()
plot(tt,a(L/2+1:end,:),"LineWidth",3)
title("fft Spectrum in the Positive and Negative Frequencies")
xlabel("f (Hz)")
ylabel("|fft(X)|")

%% 
close all;

L=1000
S = Vv1(1:L,1:10);


Y = fft(S);

YY = abs(fftshift(Y));
tt= Fs/L*(0:L/2-1);

% figure()
% plot(tt,a(L/2+1:end,:),"LineWidth",3)
% title("fft Spectrum in the Positive and Negative Frequencies")
% xlabel("f (Hz)")
% ylabel("|fft(X)|")
% 

figure()
semilogy(tt,YY(L/2+1:end,:),"LineWidth",3)
title("fft Spectrum in the Positive and Negative Frequencies")
xlabel("f (Hz)")
ylabel("|fft(X)|")
%% filter
close all;
fc = 10;
n = 4;
fs = 35;
[b,a] = butter(6,fc/(fs/2));
y = filter(b,a,S);


figure()
semilogy(tt,YY(L/2+1:end,:),"LineWidth",3)
title("fft Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")


%
yn = y./vecnorm(y,2,2);
figure()
plot(y(1:1000,1:10));hold on;

%
figure()
plot(yn(1:1000,2));hold on;
plot(Vv(1:1000,2))

%
Y = fft(y);

YY = abs(fftshift(Y));
tt= Fs/L*(0:L/2-1);

% figure()
% plot(tt,a(L/2+1:end,:),"LineWidth",3)
% title("fft Spectrum in the Positive and Negative Frequencies")
% xlabel("f (Hz)")
% ylabel("|fft(X)|")
% 

figure()
semilogy(tt,YY(L/2+1:end,:),"LineWidth",3)
title("fft Spectrum after filtering")
xlabel("f (Hz)")
ylabel("|fft(X)|")

