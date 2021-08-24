% This tutorial shows how to calculate the idealized roughness,
% harmonicity, spectral entropy, and mean spectral pitch of every member of
% a set of chords specified by pitches (e.g., semitones or cents). These
% values are returned in the table chordFeatures.
%
% Note that, in order to keep the calculation of each feature distinct,
% there is some duplication of calclations; if more than one feature is
% required, it may be useful to refactor the code to remove these
% redundancies.
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

%% Chord data (as MIDI pithes)
% An example of a small set of chords in MIDI pitches (semitones), some in
% 12-TET, some in just intonation:
chordData = [60 61 62; ... % cluster
             60 75 67; ... % open minor
             57 73 64; ... % open major
             60 63 66; ... % close diminished
             60 75.1564 67.02; ... % open just minor
             57 72.8631 64.02]; % open just major
         
%% Create table to store features
chordFeatures = table;

%% Preprocess chords data
chords = sort(chordData, 2); % put each row's entries into pitch order
nChord = size(chords, 1); % get number of chords
for chord = 1:nChord
    chordFeatures.Chord{chord} = chordData(chord, :);
end
    
chords = chords - 60; % make middle C the reference pitch of 0
chords = 100 * chords; % convert semitones to cents
pitchLimits = [min(chords(:)) max(chords(:))];
intLimits = [0 max(chords(:,end) - chords(:,1))];

fRef = 261.6256; % reference frequency -- i.e., the frequency that is 0 cents

%% Roughness: Sethares 2005
% Set parameters
pNorm = 1; % see fSetRoughness for more details 
isAve = 0; % see fSetRoughness for more details 

nHarm = 36; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);

idealRoughness = nan(nChord,1);
for chord = 1:nChord
    chord_p = chords(chord,:); 
    chord_spec_p = tmplSpec_p' + chord_p;
    chord_spec_p = chord_spec_p(:);
    chord_spec_f = fRef * 2.^(chord_spec_p/1200);
    chord_spec_w = repmat(tmplSpec_w, 1, length(chord_p));
    chord_spec_w = chord_spec_w(:);
    idealRoughness(chord) ...
        = fSetRoughness(chord_spec_f,chord_spec_w,pNorm,isAve);
end
chordFeatures.IdealRoughness = idealRoughness;

%% Harmonicity: Milne 2013
% Calculate the harmonicity using the method defined in Milne 2013, 2016,
% which is to take the maximum value of the normalized cross-correlation of
% the log-f spectra of the chord and a template harmonic complex tone.
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
limits = [0 1200*log2(nHarm)]; 

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);
templateX = expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
templateXDotProd = templateX' * templateX;                     

% Transpose chords so lowest note is 0
chords0 = chords - chords(:, 1);

limits = [0 intLimits(2) + ceil(1200*log2(nHarm))]; % enough to include all 
% harmonics of all chords' intervals
isSparse = 0;
doPlot = 0;
tol = 0;

idealHarmMilne2013 = nan(nChord,1);
for chord = 1:nChord
    chord_p = chords0(chord,:);
    chord_spec_p = tmplSpec_p' + chord_p;
    chord_spec_p = chord_spec_p(:);
    chord_spec_w = repmat(tmplSpec_w, 1, length(chord_p));
    chord_spec_w = chord_spec_w(:);
    chordX = expectationTensor(chord_spec_p, chord_spec_w, ...
        sigma, kerLen, ...
        r, isRel, isPer, limits, ...
        isSparse, doPlot, tol);
    idealHarmMilne2013(chord) ...
        = max(conv(chordX, flipud(templateX), 'full')) ...
        / sqrt((chordX'*chordX)*templateXDotProd);
end
chordFeatures.idealHarmMilne2013 = idealHarmMilne2013;

%% Harmonicity: Harrison 2020
% Calculate harmonicity as the entropy of the cross-correlation vector
% defined above. Note that this code is identical to Harmonicity: Milne
% 2013 except for the line setting wavFeatures.harmHarrison2020(wav) and
% the extra parameter isNorm.

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
limits = [0 1200*log2(nHarm)]; 

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);
templateX = expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
                              r, isRel, isPer, limits);
templateXDotProd = templateX' * templateX;                     

% Transpose chords so lowest note is 0
chords0 = chords - chords(:, 1);

limits = [0 intLimits(2) + ceil(1200*log2(nHarm))]; % enough to include all 
% harmonics of all chords' intervals
isSparse = 0;
doPlot = 0;
tol = 0;

isNorm = 1;
idealHarmHarrison2020 = nan(nChord,1);
for chord = 1:nChord
    chord_p = chords0(chord,:);
    chord_spec_p = tmplSpec_p' + chord_p;
    chord_spec_p = chord_spec_p(:);
    chord_spec_w = repmat(tmplSpec_w, 1, length(chord_p));
    chord_spec_w = chord_spec_w(:);
    chordX = expectationTensor(chord_spec_p, chord_spec_w, ...
        sigma, kerLen, ...
        r, isRel, isPer, limits, ...
        isSparse, doPlot, tol);
    idealHarmHarrison2020(chord) ...
        = histEntropy(conv(chordX, flipud(templateX), 'full') ...
        / sqrt((chordX'*chordX)*(templateXDotProd)), ...
        isNorm);
end
chordFeatures.IdealHarmHarrison2020 = idealHarmHarrison2020;

%% Harmonicity: Milne 2019 (Smit et al. 2019)
% The harmonicity values are calculated by generating the nonperiodic
% relative dyad vector or triad expectation matrix for a single harmonic
% spectrum (with roll-off rho and number of harmonics nHarm). The smoothing
% width, sigma, models uncertainty of pitch perception. From the
% expectation tensor, expectation values at each location give the
% corresponding chord's harmonicity.
%
% NB -- for triads, this calculation can require a large amount of memory (for
% the expectation matrix expTensTriad) and take a long time to calculate.
% The resolution of the expectation matrix defaults to the units of x_p (here 
% cents). The resolution can be reduced with the res parameter in order to 
% reduce computing requirements. Can be useful to save expTensTriad and 
% pValsTriad to disk for future use.
%
% Due to the length of time required to do these calculations, these parts
% are commented out by default

%{
% Parameters
nHarm = 64; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template

sigma = 12; % smoothing width
res = 2; % factor for the unit size for triad expectation array (larger values 
% reduce memory requirements); for example, if x_p has cents resolution and
% res is 2, the resulting expectation matrix has a resolution of 2 cents.

% Create the template spectrum
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^-rho;

% Get the chords' intervals (required for indexing into the expectation
% tensors)
ints = chords - chords(:, 1);
ints(:,1) = [];
limits = [0 max(ints(:))]; 

if size(chords, 2) == 2
    kerLen = 12;
    r = 2;
    isPer = 0;
    isSparse = 0;
    tol = 0;
    [expTensDyad,pValsDyad] ...
        = expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
        r, isRel, isPer, limits, ...
        isSparse, doPlot, tol);
    
    offsetDyad = 1 - pValsDyad(1);
    IdealDyadHarmMilne2019 = nan(nChord,1);
    for chord = 1:nChord
        j = ints(chord,1);
        k = ints(chord,2);
        IdealDyadHarmMilne2019(chord) = expTensDyad(j+offsetDyad);
    end
end

if size(chords, 2) == 3
    kerLen = 9; 
    r = 3;
    isRel = 1;
    isPer = 0;
    isSparse = 1;
    doPlot = 0;
    tol = 0.00001;
    
    % This calculation may be too memory and cpu intensive if limits is a wide 
    % interval and res is set too low. Useful to save the outputs 
    % (expTensTriad and pValsTriad) to disk for future use.
    [expTensTriad,pValsTriad] ...
        = expectationTensor(tmplSpec_p/res, tmplSpec_w, sigma/res, kerLen, ...
                            r, isRel, isPer, limits/res, ...
                            isSparse, doPlot, tol);
    
    offsetTriad = 1 - pValsTriad(1);
    IdealTriadHarmMilne2019 = nan(nChord,1);
    for chord = 1:nChord
        j = ints(chord,1);
        k = ints(chord,2);
        subs = [round(j/res)+offsetTriad round(k/res)+offsetTriad];
        IdealTriadHarmMilne2019(chord) = spSub4Sp(expTensTriad, subs);
    end
end
% clear expTensTriad pValsTriad expTensDyad pValsDyad
%}

%% Spectral Entropy: Milne 2017 (Milne 2017 and Smit 2019)
% Transpose chords so lowest note is 0
chords0 = chords - chords(:, 1);

% Parameters
nHarm = 64; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template
sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 12; % length of smoothing kernel in standard deviations 

chord_w = 1; % this weights all chord tones equally
isPer = 0; % if set to 1, spectral entropy will be calculated with pitch ...
% classes rather than pitches; in which case, make sure to set limits = 1200

% Variables
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^-rho;

limits = [0 intLimits(2) + ceil(1200*log2(nHarm))]; % enough to include all 
% harmonics of all chords' intervals
idealSpectralEnt = nan(nChord, 1);
for chord = 1:nChord
    chord_p = chords0(chord,:);
    idealSpectralEnt(chord) ...
        = pSetSpectralEntropy(chord_p, chord_w, ...
                              tmplSpec_p, tmplSpec_w, ...
                              sigma, kerLen, ...
                              isPer, limits);
end
chordFeatures.IdealSpectralEnt = idealSpectralEnt;

%% Mean spectral pitch
% Similar to the spectral centroid except the spectraum's weighted mean is
% taken over log-frequency instead of frequency

% Parameters
nHarm = 32; % number of harmonics in the template
rho = 1; % roll-off of harmonics in template
sigma = 12; % smoothing width (9-15 are typically good values)
kerLen = 12; % length of smoothing kernel in standard deviations 

chord_w = 1; % this weights all chord tones equally
isPer = 0; % if set to 1, spectral entropy will be calculated with pitch ...
% classes rather than pitches; in which case, make sure to set limits = 1200

% Variables
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^-rho;

idealMeanSpecPitch = nan(nChord,1);
for chord = 1:nChord
    chord_p = chords(chord,:); 
    chord_spec_p = tmplSpec_p' + chord_p;
    chord_spec_p = chord_spec_p(:);
    chord_spec_w = repmat(tmplSpec_w, 1, length(chord_p));
    chord_spec_w = chord_spec_w(:);
    idealMeanSpecPitch(chord) ...
        = (chord_spec_p'*chord_spec_w) ...
        /sum(chord_spec_w);
end
chordFeatures.IdealMeanSpecPitch = idealMeanSpecPitch;

