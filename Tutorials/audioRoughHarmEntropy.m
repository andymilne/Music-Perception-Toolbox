% This tutorial shows how to calculate the roughness, harmonicity, and 
% spectral entropy of every member of a set of audio files. 
%
% Note that after having run the code to the end of "Smooth spectra and
% find their peaks", each following section (which calculates a single
% feature) is self-contained. This means that there is some duplication of
% calclations; if more than one feature is required, it may be useful to
% refactor the code to remove these redundancies.
%
% References:
% Harrison, P. M. C. and Pearce, M. T. (2020). Simultaneous consonance in music 
%   perception and composition. Psychological Review, 127(2):216–244.
% Milne, A. J. (2013). A Computational Model of the Cognition of Tonality. PhD 
%   thesis, The Open University.
% Milne, A. J., Laney, R., and Sharp, D. B. (2016). Testing a spectral model of 
%   tonal affinity with microtonal melodies and inharmonic spectra. Musicae 
%   Scientiae, 20(4):465–494.
% Milne, A. J., Bulger, D., and Herff, S. A. (2017). Exploring the space of 
%   perfectly balanced rhythms and scales. Journal of Mathematics and Music, 11
%   (2–3):101–133.
% Sethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale. Springer Verlag, 
%   London, 2nd ed. edition.
% Smit, E. A., Milne, A. J., Dean, R. T., and Weidemann, G. (2019). Perception 
%   of affect in unfamiliar musical chords. PLOS One, 14(6):1–28.

%% Import audio files (.wav) from the folder "AudioFiles"
% Get all wav filenames
allWavsNames = dir('AudioFiles/*.wav')
numWavs = length(allWavsNames)

%% Smooth spectra and find their peaks
% Set some parameters
rampLength = 0; % ramp length (samples) of envelope applied to audio files; 
% e.g., if audio starts or ends abruptly this may produce unwanted
% high-frequency components. In such a case, a ramp-length of approximately 
% 1000 ms may be useful.
sigma = 12; % smoothing width, in cents, for spectrum (9-12 cents are typically
% good values to use).
kerLen = 12;
gKer = gaussianKernel(sigma, kerLen);
fRef = 261.6256; % reference frequecy -- i.e., the frequency that is 0 cents
doPlot = 0; % plotting?

% Make table to store time domain signals, their smoothed spectra, their peaks, 
% and other features
wavFeatures = table;
for wavNum = 1:numWavs
    audio_file = fullfile('AudioFiles', (allWavsNames(wavNum).name));
    wavFeatures.Name{wavNum} = audio_file; % name of audio file
    
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
    wavFeatures.Audio{wavNum,:} = x_t; % time-domain signal
    
    % Extract peaks from smoothed log-f spectra, where smoothing has
    % standard deviation sigma. Narrower smoothing (smaller sigma) allows
    % for closer spectral peaks to be separately resolved (partials
    % differing by about 2 * sigma + 1 cents will be separately resolved as
    % peaks). However, if the smoothing is too narrow, a single pitch with
    % some vibrato will, unhelpfully, be resolved as multiple separate
    % peaks. Hence a compromise is necessary. By eye, sigma of about 9 or
    % 12 cents typically look optimal. Also return the log-f spectrum.
    [pks_p, pks_w, sig_p] = peakPicker(x_t, Fs, sigma, fRef, doPlot);
    wavFeatures.Pks_p{wavNum,:} = pks_p; % pitches of peaks
    wavFeatures.Pks_w{wavNum,:} = pks_w; % amplitudes of peaks
    wavFeatures.Sig_p{wavNum,:} = sig_p; % unsmoothed log-f spectrum
    wavFeatures.SmoothSig{wavNum,:} ...
        = conv(sig_p, gKer, 'same'); % smoothed log-f spectrum. 
end

%% Roughness: Sethares 2005
% Set parameters (see fSetRoughness for more details)
pNorm = 1; 
isAve = 0;
for wavNum = 1:numWavs
    Pks_f = fRef * 2.^(wavFeatures.Pks_p{wavNum}/1200); % convert peaks back to
    % frequency domain
    Pks_w = wavFeatures.Pks_w{wavNum};
    wavFeatures.Roughness(wavNum) = fSetRoughness(Pks_f,Pks_w,pNorm,isAve);
end

%% Harmonicity: Milne 2013
% Calculate the harmonicity using the method defined in Milne 2013, 2016,
% which is to take the maximum value of the normalized cross-correlation of
% the log-f spectra of the signal and a template harmonic complex tone.
% This value is the cosine similarity (spectral pitch similarity) of the
% template and the signal.

% Set parameters
nHarmonics = 36; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template

sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 12; % length of smoothing kernel in standard deviations (the lower the 
% value, the more the gaussian kernel is truncated; the higher the value,
% the longer the calculation time; 6 is typically sufficient).

r = 1; % do not change
isRel = 0; % do not change
isPer = 0; % if set to 1, harmonicity will be calculated with pitch classes 
% rather than pitches; in which case, make sure to set limits = 1200
limits = ceil(1200*log2(nHarmonics) + sigma*kerLen); 

% Make the template tone
template_p = 1200*log2(1:nHarmonics);
template_w = (1:nHarmonics).^(-rho);
templateX = expectationTensor(template_p, template_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
% plot(templateX)

templateDotProd = templateX' * templateX;
for wavNum = 1:numWavs
    SmoothSigDotProd ...
        = wavFeatures.SmoothSig{wavNum}' * wavFeatures.SmoothSig{wavNum};
    wavFeatures.HarmMilne2013(wavNum) ...
        = max(conv(wavFeatures.SmoothSig{wavNum}, flipud(templateX)) ...
        / sqrt(SmoothSigDotProd*templateDotProd));
end

%% Harmonicity: Harrison 2020
% Calculate harmonicity as the entropy of the cross-correlation vector
% defined above. Note that this code is identical to Harmonicity: Milne 2013
% except for the line setting wavFeatures.harmHarrison2020(wavNum) and the extra
% parameter isNorm.

% Set parameters
nHarmonics = 36; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template
isNorm = 1; % normalize the entropy to [0,1]

sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 12; % length of smoothing kernel in standard deviations (the lower the 
% value, the more the gaussian kernel is truncated; the higher the value,
% the longer the calculation time; 6 is typically sufficient).

r = 1; % do not change
isRel = 0; % do not change
isPer = 0; % if set to 1, harmonicity will be calculated with pitch classes 
% rather than pitches; in which case, make sure to set limits = 1200
limits = ceil(1200*log2(nHarmonics) + sigma*kerLen);

% Make the template tone
template_p = 1200*log2(1:nHarmonics);
template_w = (1:nHarmonics).^(-rho);
templateX = expectationTensor(template_p, template_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
% plot(templateX)

templateDotProd = templateX' * templateX;
for wavNum = 1:numWavs
    SmoothSigDotProd ...
        = wavFeatures.SmoothSig{wavNum}' * wavFeatures.SmoothSig{wavNum};
    wavFeatures.HarmHarrison2020(wavNum) ...
        = histEntropy(conv(wavFeatures.SmoothSig{wavNum}, ...
        flipud(templateX)) / sqrt(SmoothSigDotProd*templateDotProd), ...
        isNorm);
end

%% Spectral Entropy: Milne 2017
% Calculate spectral entropy, as defined in Milne 2017 and Smit 2019.
for wavNum = 1:numWavs
    wavFeatures.SpectralEnt(wavNum) ...
        = histEntropy(wavFeatures.SmoothSig{wavNum});
end
