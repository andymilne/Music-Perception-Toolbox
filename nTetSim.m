% n-TET spectral entropy and similarity to JI
clearvars

%% Parameters
sigma = 6; % standard deviation of smoothing kernel (3-10 are good values)
kerLen = 4; 
period = 1200;

rollOff = 5/3; % roll-off of harmonics (0.5-0.7 are good values)

maxN = 100;
nHarmonics = 100;

%% Calculate spectrum
spectrum = 1:nHarmonics;

% tau = 0.7
% spec_w = exp(tau)*exp(-spectrum*tau); % an experiment with different
% parameterization of roll-off -- this gives a linear rolloff in dB, the slope
% parametrized by tau.

spec_w = spectrum.^(-rollOff);
spec_w = spec_w(:);

spec_p = period*log2(spectrum);
spec_p = spec_p(:);

%% Entropy of spectrum resulting from the sum of all scale pitches
ent = nan(maxN-1,1);
for i = 1 : maxN
    scale_p = period*(0 : i-1)/i;
    scale_p = scale_p(:);
    ent(i) = pSetSpectralEntropy(scale_p,[],spec_p,spec_w,sigma,kerLen,period);
end

figure(1)
stairs(ent)
%axis([0 100 0 1])

%% Cosine similarity of spectrum and scales. Note that the spectrum is 
% accumulated and summed over pitch to avoid duplicates. The limit, for weights
% of 1/f roll-off is 2/n.
r = 2;
isRel = 1;
isPer = 1;

if isPer == 1
    limits = period;
    spec_p = round(mod(spec_p, period));
    spec = accumarray(1 + spec_p, spec_w(:));
    [spec_p,~,spec_w] = find(spec);
    spec_p = spec_p - 1;
else
    limits = [-4800 4800];
end

sim = nan(maxN-1,1);
for i = r : maxN
    scale_p = period*(0 : i-1)/i;
    scale_p = scale_p(:);
    
    % Cosine similarity of rel dyad expectation vectors of scale and spectrum 
    sim(i) = expTensorSim(spec_p, spec_w, scale_p, 0, sigma, kerLen, ...
                           r, isRel, isPer, limits);
end

figure(2)
stairs(1-sim)
%axis([0 100 0 1])
