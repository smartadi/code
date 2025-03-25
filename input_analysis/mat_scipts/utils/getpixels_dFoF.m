function data = getpixels_dFoF(d)
% GETPIXEL_DFOF Summary of this function goes here
%   Detailed explanation goes here

pathData = append('pixel',d.td(6:7),d.td(9:10),int2str(d.en),'.mat');

source_dir ='/mnt/data/brain/';
source_dir = append(source_dir,d.mn,'/',d.td,'/',num2str(d.en));
a=dir([source_dir '/*']);
out=size(a,1);

out=out-2;
path = append(source_dir,'/frame-');

w=d.params.horizon-1;

k=d.params.kernel;
if exist(pathData) == 0
    % F = [];
    dFk=[];

    
    display('computing pixel val')
    for j = 1: length(d.params.pixel)
        f=[]
        j
        display('computing F')
        for i=1:2:out
        % for i=1:2:1000
            pathim=append(path,num2str(i-1));
            fileID = fopen(pathim,'r');
            A = fread(fileID,[560,560],'uint16')';
        
            G = mean(A(d.params.pixel(j,2)-k:d.params.pixel(j,2)+k,d.params.pixel(j,1)-k:d.params.pixel(j,1)+k),'all');
            fclose(A)
            
            f = [f,G];
        end
        % F = [F;f];
    

        display('computing df/F')
    
        fk  = [ones(1,w),f];
        fmean=[];
        dfk=[];
        fkmean=[];
        fk  = [ones(1,w),f];

        for i = 1:length(f)
        % Add an LPF filter 


            fkmean = [fkmean,mean(fk(i:i+w))];
            dfk = [dfk,(fk(i+w)-fkmean(i))/fkmean(i)*100];
    end
        dFk  = [dFk;dfk];
    end
    
    
    save(pathData,'dFk');
    data.dFk = dFk;
else
    
    data.dFk = load(pathData);

end

