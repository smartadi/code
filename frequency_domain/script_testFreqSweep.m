
s = daq.createSession('ni');
% daq.getDevices

%%
s.addAnalogOutputChannel('Dev3', 'ao0', 'Voltage');
s.addAnalogOutputChannel('Dev3', 'ao1', 'Voltage');
s.addAnalogOutputChannel('Dev3', 'ao2', 'Voltage');
% s.IsContinuous = true;
rate = 100000;
s.Rate = rate;

% extract the specified parameters
laserAmp = 1;
laserDurS = 0.1;
sweepOrImpulse = 0;

% create the waveforms for each component
if sweepOrImpulse == 0
    laser = genFreqSweep(laserAmp, rate);
    trialTimeSamps = round(30*rate);
elseif sweepOrImpulse == 1
    trialTimeS = 0.1 + laserDurS + 0.01;
    trialTimeSamps = round(trialTimeS*rate);
    laser = genLaser(laserAmp, laserDurS, trialTimeSamps, rate);
end

% no need to move galvo - set to 0 all the time
galvoX = genGalvo(0, trialTimeSamps, rate);
galvoY = genGalvo(0, trialTimeSamps, rate);

s.queueOutputData([laser galvoX galvoY]);
s.startBackground();

function waveform = genFreqSweep(laserAmp, rate)
    T = 30;
    f = rate;

    chirp = dsp.Chirp(...
        'Type', 'logarithmic', ...
        'SweepDirection', 'Unidirectional', ...
        'TargetFrequency', 15, ...
        'InitialFrequency', 0.01, ...
        'TargetTime', T, ...
        'SweepTime', T, ...
        'SamplesPerFrame', T*f, ...
        'SampleRate', f, ...
        'InitialPhase', pi/2);

    waveform = (chirp()+1)./2.*laserAmp;
end

function waveform = genLaser(laserAmp, laserDur, trialTimeSamps, rate)
    laserDurSamps = laserDur * rate;
    waveform = zeros(trialTimeSamps, 1);
%         if laserFreq == 0
    waveform(10000:10000+laserDurSamps) = laserAmp; 
%         else
%             tL = 0:1/rate:laserDur;
%             osc = laserAmp*(1+sin(2*pi*laserFreq*tL));
%             waveform(delayTimeSamps:delayTimeSamps+laserDurSamps) = osc; 
%         end


end

function waveform = genGalvo(galvoPos, trialTimeSamps, rate)
    [thisRamp, ~] = genRamp(0.001, galvoPos, rate);
    waveform = zeros(trialTimeSamps, 1) + galvoPos;
    waveform(1:numel(thisRamp)) = thisRamp;
    waveform(end-numel(thisRamp)+1:end) = fliplr(thisRamp);

end

function [ramp, t] = genRamp(duration, amp, rate)
    t = 0:1/rate:duration;
    f = 1/(duration*2);
    ramp = amp*0.5*(1-cos(t*(2*pi*f))); 
end