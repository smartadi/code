clc;
close all;
clear all;


path = '/home/nimbus/Documents/Brain/data/2024-08-04/temp/frame-';

path2 = '/home/nimbus/Documents/Brain/data/AB_0032/2023-08-08/frame-';

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

F2 = [];
for i=1:out
    pathim=append(path2,num2str(i-1));
    fileID = fopen(pathim,'r');
    A = fread(fileID,[560,560],'uint16')';
    G=[];
    for j = 1 :length(pixel)
        G = [G;A(pixel(j,1),pixel(j,2))];
    end
    i
    F2 = [F2,G];    
end



