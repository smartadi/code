clc;
close all;
clear all;

pathA = '/media/nimbus/data/brain/AL_0033/2024_07_25/1/';
pathB = '/media/nimbus/data/brain/AL_0034/2024_07_29/1/';
pathC = '/media/nimbus/data/brain/AL_0034/2024_08_19/1/';

paths.Names = [pathA;pathB;pathC];
paths.Namesid = ['pathA';'pathB';'pathC'];
[m n] = size(paths.Names);

for i =1:m
    % dir(paths.Names(i,:))
    directory_instance = dir(paths.Names(i,:));
    paths.file_names.(paths.Namesid(i,:)) = {directory_instance.name};
    paths.file_names.(paths.Namesid(i,:))(1:2)=[];
end

% 
% file_names = cell2mat(file_names);
%%
pixel = [250,260;
        250,135;
        220,330;
        360,370]
%%
a = paths.file_names.(paths.Namesid(1,:));
t = Tiff(append(paths.Names(1,:),cell2mat(a(10000))),'r');
imageData = read(t);


close all;
figure()
clims=[0,4096];
imagesc(imageData);hold on;
for i=1:4
plot(pixel(i,1),pixel(i,2),'or')
end
colorbar


%% pixelate




for k=1:m
F=[];
a = paths.file_names.(paths.Namesid(k,:));
for i=1:5000
    
    t = Tiff(append(paths.Names(k,:),cell2mat(a(i))),'r');

    % t = Tiff(append(path,cell2mat(path.file_names.a(i))),'r');
    imageData = read(t);
    G=[];

    for j = 1 :length(pixel)
        G = [G;imageData(pixel(j,2),pixel(j,1))];

    end
    i
    F= [F,G];
end
pData.(paths.Namesid(k,:))=F;


end
%%
save('rawData.mat',"paths","pData");
% load('pixel_raw.mat',"Fr");
%%
close all;
Fr = pData.pathB
Fbr = Fr(:,1:2:end);
Fvr = Fr(:,2:2:end);


figure()
plot(Fr')
a

figure
plot(Fbr')

figure()
plot(Fvr')
%%
close all;
t0=1;

T=1000;
figure()
plot(Fr(:,t0:t0+T)')


figure
plot(Fbr(:,t0:t0+T)')
title('blue')

figure()
plot(Fvr(:,t0:t0+T)')
title('violet')
%%
close all;
figure
plot(Fbr(1,t0:t0+T)')

figure
plot(Fbr(2,t0:t0+T)')

figure
plot(Fbr(3,t0:t0+T)')

figure
plot(Fbr(4,t0:t0+T)')


%% Online Mean computation

Pb=[];
Pv=[];
Pbs=[];
Pvs=[];

T2 = 2000;
t0 = 1000;
W = 50;
for i = 1:T2
    Pbs = [Pbs,sum(Fbr(:,i-W+t0:i+t0-1),2)/W];
    Pvs = [Pbs,sum(Fvr(:,i-W+t0:i+t0-1),2)/W];
    Pb = [Pb,double(Fbr(:,i+t0))- Pbs(end)];
    Pv = [Pv,double(Fvr(:,i+t0))- Pvs(end)];
end


%%
close all
figure()
plot(Pb')

figure()
plot(Pv')

figure()
plot(Pbs')

figure()
plot(Pvs')


