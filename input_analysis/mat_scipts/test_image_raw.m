clc;
close all;
clear all;


path = '/home/nimbus/Documents/Brain/data/2024-08-04/temp/frame-';

% directory_instance = dir(path);
% file_names = {directory_instance.name};
% file_names(1:2)=[];

num = 1

pathim=append(path,num2str(num));
fileID = fopen(pathim,'r');
A = fread(fileID,[560,560],'int16')';

% file_names = cell2mat(file_names);
%%
pixel = [230,220;
        360,130;
        170,330;
        196,450;
        100,200;
        200,100]
%%
source_dir ='/home/nimbus/Documents/Brain/data/2024-08-04/temp/';
a=dir([source_dir '/*'])
out=size(a,1)

out=out-2;
   
%%

% F = []
% for i=1:out
%     pathim=append(path,num2str(i-1));
%     fileID = fopen(pathim,'r');
%     A = fread(fileID,[560,560],'uint16')';
%     G=[];
%     for j = 1 :length(pixel)
%         G = [G;A(pixel(j,1),pixel(j,2))];
%     end
%     i
%     F = [F,G];
% end


%% pixelate
% Fr=[];
% 
% for i=1:length(file_names)
% %for i=1:1000
% 
%     t = Tiff(append(path,cell2mat(file_names(i))),'r');
%     imageData = read(t);
%     G=[];
% 
%     for j = 1 :length(pixel)
%         G = [G;imageData(pixel(j,1),pixel(j,2))];
% 
%     end
%     i
%     Fr = [Fr,G];
% end


Fp=F;
save('pixel_trial_0824.mat',"Fp");
%%load('pixel_raw.mat',"Fr");
%%
close all;

Fb = F(:,1:2:end);
Fv = F(:,2:2:end);


figure()
plot(F')


figure
plot(Fb')

figure()
plot(Fv')

%%
close all;
t0=5000
T=500

close all
figure()
plot(F(:,t0:t0+2*T)')


figure
plot(Fb(:,t0:t0+T)')

figure()
plot(Fv(:,t0:t0+T)')



%%

states = readmatrix('/home/nimbus/Documents/Brain/data/2024-08-04/data/states.csv');

%%
N=length(F);

sb = states(:,1:2:N);
sv = states(:,2:2:N);

figure()
plot(states(1:N))


figure()
plot(sv)

figure()
plot(sb)
%%
close all
figure
plot(sb(:,t0:t0+T)')

figure()
plot(sv(:,t0:t0+T)')

figure
plot(Fb(6,t0:t0+T)')

figure()
plot(Fv(6,t0:t0+T)')