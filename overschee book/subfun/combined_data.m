clc;
clear all;
close all;



NP = double(table2array(readtable('NP_PCA_latent_10k.csv')));
WF = double(table2array(readtable('PCA_latent_10k.csv')));
%% normalise
NP_n = NP./vecnorm(NP,2,1);
WF_n = WF./vecnorm(WF,2,1);



%%
l=10
figure()
for i=1:l
    subplot(10,2,i)
    plot(NP_n(i,1:1000),'k','Linewidth',2);hold on;
    title(num2str(i))
end
sgtitle('Latent NP')

figure()
for i=1:l
    subplot(10,2,i)
    plot(WF_n(i,1:1000),'k','Linewidth',2);hold on;
    title(num2str(i))
end
sgtitle('Latent WF')