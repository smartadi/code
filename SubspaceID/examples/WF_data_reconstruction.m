clear all;
clc;

WF_forecast = double(table2array(readtable('KPCA_forecast_10k.csv')));

U = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/WF_PCA_map.csv')));

WF_images = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_resize_10k_n100.csv')));

WF_PCA = double(table2array(readtable('/home/nimbus/Documents/aditya/neuro/code/data/data_WF_PCA_projections_small.csv')));
%%
[r t] = size(WF_forecast);

U_r = U(:,1:r);

%%

WF_re = U_r*WF_forecast;
WF = U_r*WF_PCA(1:r,:);

%%

im_re = reshape(WF_re(:,end),100,100);
im = reshape(WF(:,end),100,100);

close all;

figure()
subplot(1,2,1)
image(im,'CDataMapping','scaled')
subplot(1,2,2)
image(im_re,'CDataMapping','scaled')
