function c = scaleAffinity(scalePc,scaleWt,specPc,specWt)

persistent spectrumWt spectrumPc
global spectrumPcOut spectrumWtOut  

% NB -- this routine needs to be be updated


%SCALECONSONANCE Similarity of intervals in a scale and a given spectrum.
%
% This routine calculates the similarity of the gaussian smoothed relative
% (transpositionally invariant) dyads -- i.e., intervals -- in a scale with
% those of a spectrum (e.g., the harmonic series). For example, in the
% harmonic series there are lots of 3/2 intervals and a few 7/4 intervals,
% so a scale with similar numbers of similarly sized dyads will have higher
% overall similarity. A higher values is, therefore, more 'consonant'.
%
% c = scaleConsonance(scalePc,scaleWt,spectrumPc,spectrumWt)
%
% scalePc are the pitch classes of the scale in cents.
%
% scaleWt are the associated scale pitch weights: all ones, if there is
% only one variable entered.
%
% spectrumPc are the spectral pitches in cents: harmonics, if less than
% three variables are entered.
%
% spectrumWt are the associated spectral pitch weights: all ones, if less
% than four variables are entered.
%
% When using for tones with inharmonic spectra, choose an appropriate
% spectrumPc and set spectralizeScale to 1, below.


%% Fixed parameters (not intended to be optimized to the data)
%periodXpander = 5;

period = 1200; % size of period in pitch units
winLen = 6; % width in standard deviations of smoothing kernel
r = 2; % r-ads considered
T = 1; % transpositional invariance (1) or not (0)
doPlot = 0; % plot (1) or not (0)

%% Free parameters (can be optimized to the data)
sigma = 6; % standard deviation of smoothing kernel
rollOff = 0.6; % roll-off of harmonics
spectralizeScale = 0; % add harmonics to scale pitches (1) or not (0)
nHarmonics = 12;
noLimit = 1:nHarmonics;
fiveLimit = [1 2 3 4 5 6 8 9 10 12 15 16 18 20 24 25 27 30];

%%
if (nargin < 4) && isempty(spectrumWt)
    spectrumWt = noLimit.^(-rollOff);
elseif (nargin == 4) && isempty(spectrumWt)
    spectrumWt = specWt;
end
spectrumWtOut = spectrumWt;

if nargin < 3 && isempty(spectrumPc)
    spectrumPc = 1200*log2(noLimit);
elseif (nargin == 4) && isempty(spectrumPc)
    spectrumPc = specPc;
end
spectrumPcOut = spectrumPc;

if nargin < 2
    scaleWt = ones(length(scalePc),1);
end

if spectralizeScale == 1
    scalePc = bsxfun(@plus,scalePc',spectrumPc);
    scalePc = scalePc(:);
    scaleWt = scaleWt * spectrumWt;
    scaleWt = scaleWt(:);
end

%% Expectation tensors and cosine distance between them
scaleEmf ...
    = expectationTensor(period, sigma, winLen, ...
                        scalePc, scaleWt, ...
                        T, r, doPlot);
% figure(1)
% stairs(scaleEmf)

if exist('harmonicsEmf','var') == 0
harmonicsEmf ...
    = expectationTensor(period, sigma, winLen, ...
                        spectrumPc, spectrumWt, ...
                        T, r, doPlot);
% figure(2)
% stairs(harmonicsEmf)
end


c = 1 - scaleEmf(:)'*harmonicsEmf(:) ...
  / sqrt((harmonicsEmf(:)'*harmonicsEmf(:)) * (scaleEmf(:)'*scaleEmf(:)));

end