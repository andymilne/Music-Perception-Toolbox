function [pks_p,pks_w,absDft_f,smoothX_p,pVals] ...
        = peakPicker(x_t, Fs, sigma, thresh, fRef, doPlot)

%PEAKPICKER Convert time-domain signal to log-f spectrum, smooth, find peaks.
%
% x_t is the time domain audio audio signal
%
% Fs is its sample rate (these can be got from an audio file by [x_t,Fs] =
% audioread('audiofile.wav');])
%
% sigma is the smoothing width (standard deviation of the pitch-domain Gaussian
% kernel)
%
% fRef is the frequency assigned a value of 0 cents
%
% pks_p are the pitches of peaks (found with findpeaks) of the DFT of x_t after
% it has been translated into the pitch (log-f) domain and smoothed with a
% Gaussian kernel with standard deviation sigma
%
% pks_w are the magnitudes of the peaks with pitches pks_p
%
% absDft_f is the magnitude of the DFT of x_t (frequency domain, unsmoothed).
%
% smoothX_p is the absolute expectation vector (with sigma-width smoothing)
%
% pVals give the pitch values (cents relative to fRef) of xSmooth_p

% By Andrew J. Milne, The MARCS Institute, Western Sydney University.

%% Take the DFT of the input wav
x_t = x_t(:);
nSamples = numel(x_t);
x_f = fft(x_t)/nSamples;

%% Use "sparse sound structure" (SSS) to convert DFT to pitch (log-f) domain
% X = dft2sss(peaks,Fs,{'MP','ESP','CBR','ERBR'},'FRef',fRef);
X = dft2sss(x_f,Fs,'MP','FRef',fRef);

%% Convert to absolute pitch expectation vector
x_p = X.MP; % the pitches (cents relative to fRef) of the entries in x_w
x_w = abs(X.Phasors); % the magnitudes of the pitches in x_p
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
absDft_f = x_w;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remove almost-zero values below thresh. A sensible thresh value is about
% 60dB below the signal's max because these are likely to be perceptually
% masked (inaudible) and are possible sectral leakage artefacts
x_p(x_w < thresh) = []; 
x_w(x_w < thresh) = [];

r = 1;
isRel = 0;
isPer = 0;
limits = 1200*log2([20/fRef 20000/fRef]); % the audible frequency range
isSparse = 0;
kerLen = 12;

% These two lines are not necessary (because out-of-limits values are
% automatically removed by expectationTensor) but removing them here means
% warnings are not displayed.
x_w(x_p < limits(1) | x_p > limits(2)) = []; % remove inaudible frequencies
x_p(x_p < limits(1) | x_p > limits(2)) = []; % remove inaudible frequencies

[smoothX_p,pVals] = expectationTensor(x_p, x_w, sigma, kerLen, ...
                                      r, isRel, isPer, limits, ...
                                      isSparse, 0);
                                       
%% Extract the peaks
if doPlot==1   
    figure()
    findpeaks(smoothX_p, pVals, ...
        'MinPeakDistance', 0, ...
        'MinPeakProminence', 0.002, ...
        'Annotate', 'Extents')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[pks_w, pks_p] = findpeaks(smoothX_p, pVals, ...
                  'MinPeakDistance', 0, ...
                  'MinPeakProminence', 0.002, ...
                  'Annotate', 'Extents');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
