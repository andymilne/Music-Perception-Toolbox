function h = pSetSpectralEntropy(set_p,set_w,spec_p,spec_w,...
                                 sigma,kerLen,isPer,period)

%PSETSPECTRALENTROPY Spectral entropy of a pitch (class) multiset given a
%   spectrum.
%
%   h = pSetSpectralEntropy(scalePc, scaleWt, spectrumPc, spectrumWt): The
%   spectral entropy of the gaussian smoothed spectra produced by a pitch
%   (class) multiset (e.g., a scale or chord) where each pitch stands for a
%   spectrum of pitches given by 'spectrum'. Pitch sets with higher entropy
%   have fewer overlapping partials, hence this serves as a model of the
%   multiset's overall spectral complexity, possibly dissonance.
%
%   set_p are the pitches of the multiset in cents (pitch classes when
%   isPer==1).
%
%   set_w are the associated pitch weights: all ones, if there is only one
%   zero entered or empty.
%
%   spec_p are the spectral pitches in cents: harmonics, if less than three
%   variables are entered.
%
%   spec_w are the associated spectral weights: all ones, if less than four
%   variables are entered.
%
%   For explanations of the remaining arguments, see expectationTensor.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also: fSetRoughness

%% Fixed parameters (not intended to be optimized to the data)
if nargin < 7
    period = 1200; % size of period in pitch units
end
if nargin < 6
    kerLen = 6; % width in standard deviations of smoothing kernel
end
if nargin < 5
    sigma = 6;
end

r = 1; % r-ads considered
isRel = 0; % transpositional invariance (1) or not (0)

%% Spectralize the scale pitches
[~,set_p,set_w] = spectralize(set_p,set_w,spec_p,spec_w);
set_p = set_p(:);
set_w = set_w(:);

%% Absolute monad vector and its entropy
setExpVec = expectationTensor(set_p, set_w, sigma, kerLen, ...
                              r, isRel, isPer, period);
h = histEntropy(setExpVec(:));

end