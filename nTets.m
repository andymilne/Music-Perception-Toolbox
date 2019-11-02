% n-TET spectral entropy and similarity to JI
clearvars

%% Parameters
sigma = 6 % standard deviation of smoothing kernel (3-10 are good values)
kerLen = 4; 
isPer = 1
period = 1200;

rollOff = 5/3 % roll-off of harmonics (0.5-0.7 are good values)

maxN = 43;
nHarmonics = 100;

%% Calculate spectrum
spec_p = log2(1:nHarmonics)'*1200;
spec_w = (1:nHarmonics).^(-rollOff);
% tau = 0.5
% spec_w = exp(tau)*exp(-(1:nHarmonics)'*tau); % an experiment with different
% % parameterization of roll-off -- this gives a linear rolloff in dB, the slope
% % parametrized by tau.

isAccum = 0 % The spectrum can be accumulated and summed over pitch to avoid 
% duplicates when isPer==1 (the limit, for weights of 1/f roll-off, is 2/n).
% Set limits if isPer==0. 
if isPer == 1
    limits = period;
    if isAccum == 1
        spec_p = round(mod(spec_p, period));
        spec = accumarray(1 + spec_p(:), spec_w(:));
        [spec_p,~,spec_w] = find(spec);
        spec_p = spec_p - 1;
    end
else
    limits = [0 uBound + log2(nHarmonics)*period];
end

%% Entropy of spectrum resulting from the sum of all scale pitch classes
ent = nan(maxN-1,1);
for i = 1 : maxN
    scale_p = period*(0 : i-1)/i;
    scale_p = scale_p(:);
    ent(i) = pSetSpectralEntropy(scale_p,[],spec_p,spec_w,sigma,kerLen, ...
                                     isPer,limits);
end

figure(1)
stairs(ent,'LineWidth',2)
%axis([0 100 0 1])

%% Cosine similarity of spectrum and scales pitch classes. 
r = 2
isRel = 1

sim = nan(maxN-1,1);
for i = r : maxN
    scale_p = period*(0 : i-1)/i;
    scale_p = scale_p(:);
    
    % Cosine similarity of rel dyad expectation vectors of scale and spectrum 
    sim(i) = expTensorSim(spec_p, spec_w, scale_p, 1, sigma, kerLen, ...
                          r, isRel, isPer, limits);
end

figure(2)
stairs(1-sim,'LineWidth',2)
%axis([0 100 0 1])
