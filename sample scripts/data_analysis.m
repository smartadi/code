clc;
clear all;
close all;

path = "/run/user/1001/gvfs/smb-share:server=steinmetzsuper1.biostr.washington.edu,share=data/Subjects/ZYE_0069/2023-10-03/1";
upath = '/blue/svdSpatialComponents.npy';
upath = append(path,upath);
vpath = '/corr/svdTemporalComponents_corr.npy';
vpath = append(path,vpath);


U = readUfromNPY(upath);
V = readVfromNPY(vpath);
%%

% U = reshape(U,560*560,2000);
%% 
U = U(:,:,1:500);
%%
pixelCorrelationViewerSVD(U, V);