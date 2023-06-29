fr = 100; % Hz
dt = 1/1000; % s
nBins = 10; % 10 ms spike train
myPoissonSpikeTrain = rand(1, nBins) < fr*dt;


fr = 100; % Hz
dt = 1/1000; % s
nBins = 10; % 10 ms spike train
nTrials = 20; % number of simulations
spikeMat = rand(nTrials, nBins) < fr*dt;

[spikeMat, tVec] = poissonSpikeGen(30, 1, 20);

[spikeMat, tVec] = poissonSpikeGen(30, 1, 20);
plotRaster(spikeMat, tVec*1000);
xlabel('Time (ms)');
ylabel('Trial Number');


% simulate the baseline period
[spikeMat_base, tVec_base] = poissonSpikeGen(6, 0.5, 20);
tVec_base = (tVec_base - tVec_base(end))*1000 - 1;
 
% simulate the stimulus period
[spikeMat_stim, tVec_stim] = poissonSpikeGen(30, 1, 20);
tVec_stim = tVec_stim*1000;
 
% put the baseline and stimulus periods together
spikeMat = [spikeMat_base spikeMat_stim];
tVec = [tVec_base tVec_stim];
 
% plot the raster and mark stimulus onset
plotRaster(spikeMat, tVec);
hold all;
plot([0 0], [0 size(spikeMat, 1)+1]);
 
% label the axes
xlabel('Time (ms)');
ylabel('Trial number');




function [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials)
dt = 1/1000; % s
nBins = floor(tSim/dt);
spikeMat = rand(nTrials, nBins) < fr*dt;
tVec = 0:dt:tSim-dt;
end

function [] = plotRaster(spikeMat, tVec)
hold all;
for trialCount = 1:size(spikeMat,1)
    spikePos = tVec(spikeMat(trialCount, :));
    for spikeCount = 1:length(spikePos)
        plot([spikePos(spikeCount) spikePos(spikeCount)], ...
            [trialCount-0.4 trialCount+0.4], 'k');
    end
end
ylim([0 size(spikeMat, 1)+1]);
end