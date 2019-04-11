function [signal,pVals] = noiseSignal(x_t, Fs, sizeMed, noiseFactor, sigma, ...
                                      fRef, doPlot)

%NONLINDPS Auditory nonlinear distortion products.
%
%   Author: Andrew J. Milne
%   Revision: 1.00
%   Date: 2017/11/16
%

% Noise signal splitter 
% This routine accepts an audio file and splits it into noise and signal
% components. x_t is the time domain audio audio signal, Fs is 
% its sample rate (these can be got from an audio file by 
% [x_t,Fs] = audioread('audiofile.wav');]). sizeMed sets the
% number of frequency bins over which the median of the frequency domain
% signal is calculated. The noiseFactor multplies the median signal to 
% set an appropriate signal-noise threshold (i.e., all frequency bins with 
% magnitude greater than are candidate partials).

%% Parameters
if nargin < 3
    sizeMed = 40;
    noiseFactor = 3.6;
    sigma = 6;
    fRef = 261.6256;
    doPlot = 0;
elseif nargin < 4
    noiseFactor = 3.6;
    sigma = 6;
    fRef = 261.6256;
    doPlot = 0;
elseif nargin < 5
    sigma = 6;
    fRef = 261.6256;
    doPlot = 0;
elseif nargin < 6
    fRef = 261.6256;
    doPlot = 0;
elseif nargin < 7
    doPlot = 0;
end
winLen = 6;

%% Checks
if min(size(x_t)) > 1
    error('audio input must be mono')
end

%% Take the DFT of the input wav
x_t = x_t(:);
nSamples = numel(x_t);

x_f = fft(x_t)/nSamples;
abs_x_f = abs(x_f);

if doPlot==1
    figure(1)
    stairs(abs_x_f)
end

%% Calculate the DFT
x_f = fft(x_t)/nSamples;
abs_x_f = abs(x_f);

if doPlot==1
    figure(2)
    stairs(abs_x_f)
end

%% Median filter
noiseFloor = medfilt1(abs_x_f,sizeMed);

if doPlot==1
    figure(3)
    stairs(noiseFloor)
end
    
%% Extract peaks
aboveNoise = abs_x_f;
aboveNoise(aboveNoise < noiseFloor*noiseFactor) = 0;

if doPlot==1
    figure(4)
    stairs(aboveNoise)
end

% OR???
% aboveNoise = abs_x_f - noiseFloor*noiseFactor;
% aboveNoise(aboveNoise < 0) = 0;
% 
% if doPlot==1
%     figure(5)
%     stairs(aboveNoise)
% end

%% Isolate peaks: if there are consecutive bins above the 
% noisefloor, choose only the maximum
% find consecutive >0 bins
% {
aboveNoiseBinary = [0; aboveNoise; 0];
aboveNoiseBinary(aboveNoiseBinary > 0) = 1;
aboveNoiseBinDiff = diff(aboveNoiseBinary);
starts = find(aboveNoiseBinDiff == 1);
ends = find(aboveNoiseBinDiff == -1);
nStarts = size(starts,1);

peakIdx = nan(nStarts,1);
for i = 1:nStarts
   [~,idx] = max(aboveNoise(starts(i):ends(i)-1));
   peakIdx(i) = idx+starts(i)-1;
end

% Remove nonpeaks
peaks = zeros(size(aboveNoise,1),1);
peaks(peakIdx) = aboveNoise(peakIdx);

if doPlot==1
    figure(6)
    stairs(peaks)
end
%}

%% Alternative method sweep through and take highest peak in window 
% (but it is rather slow)
%{
rectWinLen = 5;
peakIdx = zeros(nSamples);
for i = 1 : nSamples-rectWinLen
    [~,idx] = max(aboveNoise(i : i+rectWinLen));
    peakIdx(i,i+idx-1) = 1;
end
sumPeakIdx = sum(peakIdx);
sumPeakIdx(sumPeakIdx > rectWinLen) = 1;
sumPeakIdx = logical(sumPeakIdx);

% Remove nonpeaks
peaks = zeros(size(aboveNoise,1),1);
peaks(sumPeakIdx) = aboveNoise(sumPeakIdx);

if doPlot==1
    figure(7)
    stairs(peaks)
end
%}

%% Convert peaks to "sparse sound structure" (SSS)
% X = dft2sss(peaks,Fs,{'MP','ESP','CBR','ERBR'},'FRef',fRef);
X = dft2sss(peaks,Fs,'MP','FRef',fRef);

if doPlot==1
    figure(10)
    stairs(X.F,abs(X.Phasors))

    figure(11)
    stairs(X.MP,abs(X.Phasors))

%     figure(12)
%     stairs(X.ESP,abs(X.Phasors))
% 
%     figure(13)
%     stairs(X.CBR,abs(X.Phasors))
% 
%     figure(14)
%     stairs(X.ERBR,abs(X.Phasors))
end

%% Convert to absolute pitch class expectation vector for musical pitch
x_p = X.MP;
x_w = abs(X.Phasors);
x_p(x_w < 0.0001) = []; % remove almost zero values
x_w(x_w < 0.0001) = []; % remove almost zero values
isRel = 0;
r = 1;
isPer = 0;
limits = 1200*log2([20/fRef 20000/fRef]);
isSparse = 0;

% resolution = 1;
% period = ceil(X.MP(end));

[signal,pVals] = expectationTensor(x_p, x_w, sigma, winLen, ...
                                   r, isRel, isPer, limits, ...
                                   isSparse, doPlot);

end
