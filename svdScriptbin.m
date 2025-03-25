%% TO BE USED WITH Aditya's exeriments

% addpath(genpath(fullfile('C:\Users\SteinmetzLab\Documents\github\')));
% addpath(genpath(fullfile('C:\Users\SteinmetzLab\Documents\MATLAB\Github')));
%% 
% pause(11400);

allPaths = {};


allPaths{end+1} = 'AL_0033\2025-01-29\1';


for ii=1:size(allPaths,2)
    close all;
    try
        currentPath = allPaths{ii};
        fprintf(1, 'start %s\n', currentPath);
        loadAndSVDfbin(currentPath);
        fprintf(1, 'Done with %s\n', currentPath);
    catch me
        fprintf(1, 'Error with %s\n', currentPath);
        disp(me)
    end

end