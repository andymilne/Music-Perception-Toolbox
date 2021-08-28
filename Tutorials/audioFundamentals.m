% This tutorial shows how to estimate the weights of all possible
% fundamentals (virtual pitches) of every member of a set of audio files.
% (See audRoughHarmEntropy for how the vector of fundamental weights
% calculated here can be used to calculate the harmonicity of the audio
% files.)
%
% References:
% Milne, A. J. (2013). A Computational Model of the Cognition of Tonality. PhD 
%   thesis, The Open University.
% Milne, A. J., Laney, R., and Sharp, D. B. (2016). Testing a spectral model of 
%   tonal affinity with microtonal melodies and inharmonic spectra. Musicae 
%   Scientiae, 20(4):465–494.

%% Import audio files (.wav) from the folder "AudioFiles"
% Get all wav filenames
allWavsNames = dir('AudioFiles/*.wav')
nWav = length(allWavsNames)%% Fundamentals
% Use the cross correlation with the harmonic template to get the weights of 
% possible fundamentals (virtual pitches)

%% Smooth spectra and find their peaks
% Set some parameters
rampLength = 0; % ramp length (samples) of envelope applied to audio files; 
% e.g., if audio starts or ends abruptly this may produce unwanted
% high-frequency components. In such a case, a ramp-length of approximately 
% 1000 ms may be useful.
sigma = 12; % smoothing width, in cents, for spectrum (9-12 cents are typically
% good values to use).
fRef = 261.6256; % reference frequency -- i.e., the frequency that is 0 cents
doPlot = 0; % plotting?

% Make table to store time domain signals, their smoothed spectra, their peaks, 
% and other features
wavFeatures = table;
name = cell(nWav,1);
audio = cell(nWav,1);
allPks_p = cell(nWav,1);
allPks_w = cell(nWav,1);
allAbsDFT_f = cell(nWav,1);
allSmoothX_p = cell(nWav,1);
allPVals = cell(nWav,1);
for wav = 1:nWav
    audio_file = fullfile('AudioFiles', (allWavsNames(wav).name));
    name{wav} = audio_file; % name of audio file
    
    [x_t, Fs] = audioread(char(audio_file));
    x_t = sum(x_t, 2)/2; % Collapse stereo to mono
            
    % Envelope audio with ramps
    if rampLength > 0
        envelope = ones(1,length(x3_t));
        envelope(1 : rampLength+1) = 0 : 1/rampLength : 1;
        envelope(end-rampLength : end) = fliplr(0 : 1/rampLength : 1);
        envelope = envelope';
        x_t = x_t .* envelope;
    end
    audio{wav,:} = x_t; % time-domain signal
    
    % Extract peaks from smoothed log-f spectra, where smoothing has
    % standard deviation sigma. Narrower smoothing (smaller sigma) allows
    % for closer spectral peaks to be separately resolved (partials
    % differing by about 2 * sigma + 1 cents will be separately resolved as
    % peaks). However, if the smoothing is too narrow, a single pitch with
    % some vibrato will, unhelpfully, be resolved as multiple separate
    % peaks. Hence a compromise is necessary. By eye, sigma of about 9 or
    % 12 cents typically look optimal. Also return the log-f spectrum.
    [pks_p, pks_w, absDFT_f, xSmooth_p, pVals] ...
        = peakPicker(x_t, Fs, sigma, fRef, doPlot);
    allPks_p{wav,:} = pks_p; % pitches of peaks
    allPks_w{wav,:} = pks_w; % amplitudes of peaks
    allAbsDFT_f{wav,:} = absDFT_f; % unsmoothed freq spectrum
    allSmoothX_p{wav,:} = xSmooth_p; % smoothed log-f spectrum
    allPVals{wav,:} = pVals; % pitches in xSmooth_p
%     allSmoothX_p{wav,:} ...
%         = conv(sig_p, gKer, 'same'); % smoothed log-f spectrum. 
end
wavFeatures.Name = name;
wavFeatures.Audio = audio;
wavFeatures.Pks_p = allPks_p;
wavFeatures.Pks_w = allPks_w;
wavFeatures.absDFT_f = allAbsDFT_f;
wavFeatures.SmoothX_p = allSmoothX_p;
wavFeatures.PVals = allPVals;

%% Fundamentals
% Use the cross correlation with the harmonic template to get the weights of 
% possible fundamentals (virtual pitches).

% Set parameters
nHarm = 36; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template

sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 6; % length of smoothing kernel in standard deviations (the lower the 
% value, the more the gaussian kernel is truncated; the higher the value,
% the longer the calculation time; 6 is typically sufficient).

r = 1; % do not change
isRel = 0; % do not change
isPer = 0; % if set to 1, harmonicity will be calculated with pitch classes 
% rather than pitches; in which case, make sure to set limits = 1200
limits = ceil(1200*log2(nHarm) + sigma*kerLen); 

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);
templateX = expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
templateDotProd = templateX' * templateX;

offset = floor(ceil(sigma*kerLen)/2); % for calculating pitch values pValsExt 
% in cross-correlation vector

audFundMilne2021 = cell(nWav,1);
pValsExt = cell(nWav,1);
for wav = 1:nWav
    SmoothX_pDotProd ...
        = wavFeatures.SmoothX_p{wav}' * wavFeatures.SmoothX_p{wav};
    audFundMilne2021{wav} ...
        = conv(wavFeatures.SmoothX_p{wav}, flipud(templateX)) ...
        / sqrt(SmoothX_pDotProd*templateDotProd);
    pValsExt{wav} ...
        = (allPVals{wav}(1) - length(templateX) + 1 : allPVals{wav}(end))' ...
        + offset;
end
wavFeatures.AudFundMilne2021 = [pValsExt audFundMilne2021];
% The first column of AudFundMilne2021 gives the pitches in cents relative
% to middle C, the second column shows the weight given to that pitch being
% a fundamental (or virtual pitch) of the spectrum.

% Example plot with MIDI pitches on the horizontal axis 
figure(1)
wavToPlot = 4;
plot(wavFeatures.AudFundMilne2021{wavToPlot,1}/100 + 60, ... % MIDI piches
     wavFeatures.AudFundMilne2021{wavToPlot,2})
xticks([0:4:130])
axis([0, 128, 0, 1.1*max(wavFeatures.AudFundMilne2021{wavToPlot,2})])
