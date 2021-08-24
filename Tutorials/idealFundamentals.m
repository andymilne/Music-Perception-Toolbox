% This tutorial shows how to estimate the weights of all possible
% fundamentals (virtual pitches) of every member of a set of chords (here
% specified by MIDI pitches). (See idealRoughHarmEntropy for how the vector
% of fundamental weights calculated here can be used to calculate the
% harmonicity of the audio files.)
%
% References:
% Milne, A. J. (2013). A Computational Model of the Cognition of Tonality. PhD 
%   thesis, The Open University.
% Milne, A. J., Laney, R., and Sharp, D. B. (2016). Testing a spectral model of 
%   tonal affinity with microtonal melodies and inharmonic spectra. Musicae 
%   Scientiae, 20(4):465–494.

%% Chord data (as MIDI pithes)
% An example of a small set of chords in MIDI pitches (semitones), some in
% 12-TET, some in just intonation:
chordData = [60 61 62; ... % cluster
             60 75 67; ... % open C minor
             57 73 64; ... % open A major
             60 63 66; ... % close C diminished
             60 75.1564 67.02; ... % open just C minor
             57 72.8631 64.02]; % open just A major
                  
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

%% Fundamentals
% Use the cross correlation with the harmonic template to get the weights of 
% possible fundamentals (virtual pitches).

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
limits = [0 ceil(1200*log2(nHarm))]; 

% Make the template tone
tmplSpec_p = 1200*log2(1:nHarm);
tmplSpec_w = (1:nHarm).^(-rho);
[templateX,pValTemp] = ...
    expectationTensor(tmplSpec_p, tmplSpec_w, sigma, kerLen, ...
    r, isRel, isPer, limits);
templateXDotProd = templateX' * templateX;                     

offset = floor(ceil(sigma*kerLen)/2); % for calculating pitch values pValsExt 
% in cross-correlation vector

limits = [pitchLimits(1) pitchLimits(2) + ceil(1200*log2(nHarm))]; % enough to 
% include all harmonics of all chords' intervals
isSparse = 0;
doPlot = 0;
tol = 0;

idealFundMilne2021 = cell(nChord,1);
pValsExt = cell(nChord,1);
for chord = 1:nChord
    chord_p = chords(chord,:);
    chord_spec_p = tmplSpec_p' + chord_p;
    chord_spec_p = chord_spec_p(:);
    allPVals = sort(chord_spec_p);
    chord_spec_w = repmat(tmplSpec_w, 1, length(chord_p));
    chord_spec_w = chord_spec_w(:);
    [chordX,pVals] = expectationTensor(chord_spec_p, chord_spec_w, ...
        sigma, kerLen, ...
        r, isRel, isPer, limits, ...
        isSparse, doPlot, tol);
    idealFundMilne2021{chord} ...
        = conv(chordX, flipud(templateX), 'full') ...
        / sqrt((chordX'*chordX)*templateXDotProd);
    pValsExt{chord} ...
        = (pVals(1) - length(templateX) + 1 : pVals(end))' + offset;
end
chordFeatures.IdealFundMilne2021 = [pValsExt idealFundMilne2021];
% The first column of IdealFundMilne2021 gives the pitches in cents relative
% to middle C, the second column shows the weight given to that pitch being
% a fundamental (or virtual pitch) of the spectrum

% Example plot
figure(1)
chordToPlot = 3;
plot(chordFeatures.IdealFundMilne2021{chordToPlot,1}, ...
    chordFeatures.IdealFundMilne2021{chordToPlot,2})
