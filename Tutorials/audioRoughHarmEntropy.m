% This tutorial shows how to calculate the roughness, harmonicity, spectral
% entropy, and mean spectral pitch of every member of a set of audio files.
% These values are returned in the table wavFeatures.
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
nWav = length(allWavsNames)

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
end
wavFeatures.Name = name;
wavFeatures.Audio = audio;
wavFeatures.Pks_p = allPks_p;
wavFeatures.Pks_w = allPks_w;
wavFeatures.absDFT_f = allAbsDFT_f;
wavFeatures.SmoothX_p = allSmoothX_p;
wavFeatures.PVals = allPVals;

%% Roughness: Sethares 2005
% Set parameters (see fSetRoughness for more details)
pNorm = 1; 
isAve = 0;
audRoughness = nan(nWav,1);
for wav = 1:nWav
    Pks_f = fRef * 2.^(wavFeatures.Pks_p{wav}/1200); % convert peaks back to
    % frequency domain
    Pks_w = wavFeatures.Pks_w{wav};
    audRoughness(wav) = fSetRoughness(Pks_f,Pks_w,pNorm,isAve);
end
wavFeatures.AudRoughness = audRoughness;

%% Harmonicity: Milne 2013
% Calculate the harmonicity using the method defined in Milne 2013, 2016,
% which is to take the maximum value of the normalized cross-correlation of
% the log-f spectra of the signal and a template harmonic complex tone.
% This value is the cosine similarity (spectral pitch similarity) of the
% template and the signal.

% Set parameters
nHarm = 36; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template

sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 12; % length of smoothing kernel in standard deviations (the lower the 
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
                          
% plot(templateX)

templateDotProd = templateX' * templateX;
audHarmMilne2013 = nan(nWav,1);
for wav = 1:nWav
    SmoothX_pDotProd ...
        = wavFeatures.SmoothX_p{wav}' * wavFeatures.SmoothX_p{wav};
    audHarmMilne2013(wav) ...
        = max(conv(wavFeatures.SmoothX_p{wav}, flipud(templateX), 'full')) ...
        / sqrt(SmoothX_pDotProd*templateDotProd);
end
wavFeatures.AudHarmMilne2013 = audHarmMilne2013;

%% Harmonicity: Harrison 2020
% Calculate harmonicity as the entropy of the cross-correlation vector
% defined above. Note that this code is identical to Harmonicity: Milne
% 2013 except for the line setting wavFeatures.harmHarrison2020(wav) and
% the extra parameter isNorm.

% Set parameters
nHarm = 36; % number of harmonics in the template
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
limits = ceil(1200*log2(nHarm) + sigma*kerLen);

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);
templateX = expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
% plot(templateX)

templateDotProd = templateX' * templateX;
audHarmHarrison2020 = nan(nWav,1);
for wav = 1:nWav
    SmoothX_pDotProd ...
        = wavFeatures.SmoothX_p{wav}' * wavFeatures.SmoothX_p{wav};
    audHarmHarrison2020(wav) ...
        = histEntropy(conv(wavFeatures.SmoothX_p{wav}, ...
        flipud(templateX), 'full') ...
        / sqrt(SmoothX_pDotProd*templateDotProd), ...
        isNorm);
end
wavFeatures.AudHarmHarrison2020 = audHarmHarrison2020;

%% Spectral Entropy: Milne 2017
% Calculate spectral entropy, as defined in Milne 2017 and Smit 2019.
audSpectralEnt = nan(nWav,1);
for wav = 1:nWav
    audSpectralEnt(wav) ...
        = histEntropy(wavFeatures.SmoothX_p{wav});
end
wavFeatures.AudSpectralEnt = audSpectralEnt;

%% Mean spectral pitch
% Similar to the spectral centroid except the spectrum's weighted mean is
% taken over log-frequency instead of frequency

audMeanSpecPitch = nan(nWav,1);
for wav = 1:nWav
    audMeanSpecPitch(wav) ...
        = (wavFeatures.PVals{wav}'*wavFeatures.SmoothX_p{wav}) ...
        /sum(wavFeatures.SmoothX_p{wav});
end
wavFeatures.AudMeanSpecPitch = audMeanSpecPitch;

