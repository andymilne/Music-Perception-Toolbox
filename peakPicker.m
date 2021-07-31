function [pks_p,pks_w,sig_p] = peakPicker(x_t, Fs, sigma, ...
                                          fRef, doPlot)

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
% sig_p is the DFT of x_t translated into the pitch (log-f) domain.

% By Andrew J. Milne, The MARCS Institute, Western Sydney University.

%% Take the DFT of the input wav
x_t = x_t(:);
nSamples = numel(x_t);
x_f = fft(x_t)/nSamples;

%% Use "sparse sound structure" (SSS) to convert DFT to pitch (log-f) domain
% X = dft2sss(peaks,Fs,{'MP','ESP','CBR','ERBR'},'FRef',fRef);
X = dft2sss(x_f,Fs,'MP','FRef',fRef);

%% Convert to absolute pitch expectation vector
x_p = X.MP;
x_w = abs(X.Phasors);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig_p = x_w;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_p(x_w < 0.0001) = []; % remove almost zero values
x_w(x_w < 0.0001) = []; % remove almost zero values
r = 1;
isRel = 0;
isPer = 0;
limits = 1200*log2([20/fRef 20000/fRef]);
isSparse = 0;
kerLen = 12;
[xSmooth_p,xSmooth_wp] = expectationTensor(x_p, x_w, sigma, kerLen, ...
                                           r, isRel, isPer, limits, ...
                                           isSparse, doPlot);

%% Extract the peaks
if doPlot==1   
    figure()
    findpeaks(xSmooth_p, xSmooth_wp, ...
        'MinPeakDistance', 0, ...
        'MinPeakProminence', 0.002, ...
        'Annotate', 'Extents')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[pks_w, pks_p] = findpeaks(xSmooth_p, xSmooth_wp, ...
                  'MinPeakDistance', 0, ...
                  'MinPeakProminence', 0.002, ...
                  'Annotate', 'Extents');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
