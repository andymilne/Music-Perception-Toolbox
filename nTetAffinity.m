% n-TET spectral entropy
clear all

limits = 1200;
kerLen = 6; 
isRel = 0;
isPer = 1;
r = 1;

sigma = 6; % standard deviation of smoothing kernel (6-10 are good values)
rollOff = 0.6; % roll-off of harmonics (0.5-0.7 are good values)
nHarmonics = 12;

noLimit = (1:nHarmonics);
fiveLimit = [1 2 3 4 5 6 8 9 10 12 15 16 18 20 24 25 27 30];
spectrum = noLimit;

specWt = spectrum.^(-rollOff);
specPc = 1200*log2(spectrum);
specPc = specPc(:);

maxN = 100;
d = nan(maxN-1,1);
for i = 2:maxN
    scalePc = 1200*(0:(i-1))/i;
    scalePc = scalePc(:);
    % spectralize
    scaleWt = ones(length(scalePc),1);
    scaleWt = scaleWt * specWt;
    scaleWt = scaleWt(:);
    %stop
    scalePc = bsxfun(@plus,scalePc,specPc');
    scalePc = scalePc(:);
    
    
    % Expectation tensor and its entropy
    scaleEmf ...
        = expectationTensor(scalePc, scaleWt, sigma, kerLen, ...
                            r, isRel, isPer, limits);
    
    
    d(i) = histEntropy(scaleEmf(:))/log2(i);
end

figure(4)
stairs(d)