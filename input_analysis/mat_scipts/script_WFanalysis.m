% Read and analyze data
clear all
close all;
clc;

%% raw WF data random experiment

load('pixel_raw.mat')
%%
Fbr = Fr(:,1:2:end);
Fvr = Fr(:,2:2:end);

%% data from svd (with and w/o mean) for the same dataset

load('pixel_real.mat')
Fim = F;

%%
close all

figure()
% plot(F(1,:));hold on;
plot(F2(1,1:end));hold on;
% plot(Fbr(1,:));hold on;
% plot(Fvr(1,:));hold on;
%% data from feedback exp

load('pixel_var.mat')

Fb = F(:,1:2:end);
Fv = F(:,2:2:end);




%%
% close all


figure()
plot(F(1,1:1000));hold on;



figure()
plot(Fb(1,1:500));hold on;
plot(Fv(1,1:500));hold on;