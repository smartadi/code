clc;
close all;
clear all;

path = "/run/user/1000/gvfs/smb-share:server=sahale.biostr.washington.edu,share=data/Subjects/test/2024-08-27/6/";

% path = '/home/nimbus/Documents/Brain/data/2024-08-04/temp/frame-';
% 
% path2 = '/home/nimbus/Documents/Brain/data/AB_0032/2023-08-08/frame-';

% directory_instance = dir(path);
% file_names = {directory_instance.name};
% file_names(1:2)=[];

num = 1
%%

WF = readNPY(path+'widefieldExposure.raw.npy');
%%
WF_times = length(find(WF(2:end)>1 & WF(1:end-1)<=1));
% ds = find(diff([0;stimTimes])>0.05);
% stimStarts = stimTimes(ds);
% stimEnds = stimTimes(ds(2:end)-1); 
%%
path = '/home/nimbus/Documents/Brain/data/2024-08-27/img_bin';


% pathim=append(path,num2str(num));
fileID = fopen(path,'r');
A = fread(fileID,[560*560, WF_times],'*uint16');
%%
i=1000
img = reshape(A(:,i),[560,560]);

figure()
imagesc(img)
%%
% file_names = cell2mat(file_names);
% %%
% pixel = [230,220;
%         360,130;
%         170,330;
%         196,450;
%         100,200;
%         200,100]
% %%
% % source_dir ='/home/nimbus/Documents/Brain/data/2024-08-04/temp/';
% % a=dir([source_dir '/*'])
% % out=size(a,1)
% % 
% % out=out-2;   
% %%
% 
% F2 = []
% for i=1:out
%     pathim=append(path2,num2str(i-1));
%     fileID = fopen(pathim,'r');
%     A = fread(fileID,[560,560],'uint16')';
%     G=[];
%     for j = 1 :length(pixel)
%         G = [G;A(pixel(j,1),pixel(j,2))];
%     end
%     i
%     F2 = [F2,G];
% end
% 
% 
% %% pixelate
% % Fr=[];
% % 
% % for i=1:length(file_names)
% % %for i=1:1000
% % 
% %     t = Tiff(append(path,cell2mat(file_names(i))),'r');
% %     imageData = read(t);
% %     G=[];
% % 
% %     for j = 1 :length(pixel)
% %         G = [G;imageData(pixel(j,1),pixel(j,2))];
% % 
% %     end
% %     i
% %     Fr = [Fr,G];
% % end
% 
% 
% Fp2=F2;
% save('pixel_trial_080824.mat',"Fp2");
% %%load('pixel_raw.mat',"Fr");
% %%
% close all;
% 
% Fb = F2(:,1:2:end);
% Fv = F2(:,2:2:end);
% 
% 
% figure()
% plot(F2')
% 
% 
% figure
% plot(Fb')
% 
% figure()
% plot(Fv')
% 
% %%
% close all;
% t0=5000
% T=500
% 
% close all
% figure()
% plot(F2(:,t0:t0+2*T)')
% 
% 
% figure
% plot(Fb(:,t0:t0+T)')
% 
% figure()
% plot(Fv(:,t0:t0+T)')
% 
% 
% 
% %%
% 
% states = readmatrix('/home/nimbus/Documents/Brain/data/AB_0032/2023-08-08/states.csv');
% 
% %%
% close all
% N=length(F2);
% 
% sb = states(:,1:2:N);
% sv = states(:,2:2:N);
% 
% figure()
% plot(states(1:N))
% 
% %%
% close all
% figure()
% plot(sv)
% 
% figure()
% plot(sb)
% %%
% close all
% figure
% plot(sb(:,t0:t0+T)')
% 
% figure
% plot(Fb(6,t0:t0+T)')
% 
% figure()
% plot(Fv(6,t0:t0+T)')