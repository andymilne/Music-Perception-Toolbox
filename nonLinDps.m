function A = nonLinDps(polyCoeffs,loPass,isEven,isOdd,X_f,X_w,doPlot)

%NONLINDPS Spectra resulting from polynomial waveshaping of spectra.
%
%   polyCoeffs - if isOdd==0, these are the coefficients of even terms 2, 4, 6,
%   ... of the polynomial waveshaper; if isOdd==1, these are the coefficients
%   of the odd terms 1, 3, 5, ..., of the polynomial waveshaper. All
%   non-specified coefficients are set to zero.
%
%   X_f = a matrix, each of whose rows is a set of frequencies to be 
%   independently waveshaped in the time domain.
%
%   X_w = the complex values (representing the magnitudes and phases) of the 
%   frequencies in X_f
%
%   doPlot = 0 or 1 - do or do not make a plot of the resulting spectrum.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University

persistent lenPosSpec spectra lenSpec nSpectra maxInFreq inWavePowTens

if isOdd==0 && isEven==0
    error('One or both of isOdd and isEve must be 1.')
end

if nargin < 7
    doPlot = 0;
end

%% Build spectral weights vector (alpha)
nCoeffs = length(polyCoeffs);

%% Make a weighted spectral indicator matrix -- each row for each spectrum,
% with zero-padding. As few zeros as possible are added -- sufficient to
% ensure all distortion products are below the Nyquist and additionally
% padded to make length a power of two
if isempty(nSpectra)
    nSpectra = size(X_f,1);
end

if isempty(maxInFreq)
    maxInFreq = max(max(X_f));
end

if isempty(lenPosSpec)
    maxOutFreq = maxInFreq*nCoeffs;
    lenPosSpec = 2^nextpow2(maxOutFreq+1);
end

if isempty(lenSpec)
    lenSpec = 2*lenPosSpec;
end

%% Add negative frequencies to input spectra, put into matrix "spectra"
% (spectrum x frequency), normalize
if isempty(spectra)
    spectra = zeros(nSpectra, lenSpec);
    for spectrumNo = 1:nSpectra
        X_f(spectrumNo,:)
        X_w(spectrumNo,:)
        spectra(spectrumNo, 1:lenPosSpec) ...
            = weightedInd(X_f(spectrumNo,:),X_w(spectrumNo,:),lenPosSpec);
        % Add negative frequencies
        spectra(spectrumNo,1:lenSpec) ...
            = [spectra(spectrumNo,1:lenPosSpec) 0 ...
            fliplr(conj(spectra(spectrumNo,2:lenPosSpec)))];
    end
end

%% Convert spectra to waveforms, normalize, calculate their powers, put into
% tensor (power x freq x spectrum)
if isempty(inWavePowTens)
    waveMatr = ifft(spectra')';
    waveMatr = real(waveMatr);
    waveMatr = waveMatr./max(abs(waveMatr),[],2); % normalize so max abs value 
    % is 1
    if isOdd==0 && isEven==1
        powInd = permute(0 : 2 : 2*nCoeffs-1,[1 3 2]);
    elseif isOdd==1 && isEven==0
        powInd = permute(1 : 2 : 2*nCoeffs-1,[1 3 2]);
    elseif isOdd==1 && isEven==1
        powInd = permute(0 : nCoeffs-1,[1 3 2]);
    end
    inWavePowTens = waveMatr.^powInd;
    inWavePowTens = permute(inWavePowTens,[3,2,1]);
end

%% Calculate waveform resulting from the polynomial coefficients
inWavePowMatr = reshape(inWavePowTens,[nCoeffs lenSpec*nSpectra]);
dpWave = polyCoeffs*inWavePowMatr;
dpWave = reshape(dpWave, [lenSpec nSpectra]);

%% Convert waveform to spectrum SHOULD I USE ABS (maybe better to return
% complex number and let it be made aboslute outside this routine)?
dpSpectrum = abs(fft(dpWave))/lenSpec;
dpSpectrum = dpSpectrum';

%% Normalize dpSpectrum
% For interpretability, the non-DC bins of the spectrum are multiplied by
% 2, so that a magnitude of 1 (rather than the conventional 0.5)
% corresponds to a sinusoid ranging from -1 to 1 (full digital range). The
% DC and Nyquist components do not need this doubling, hence the former
% is not doubled; there can be no frequency at the Nyquist due to the
% earlier zero-padding of the input spectrum vector.
normalizer = [1 2*ones(1,lenSpec - 1)];
dpSpectrum = dpSpectrum.*normalizer;

%% Remove negative frequencies
dpSpectrum = dpSpectrum(:,1:lenPosSpec);

%% FIND BETTER WAY TO REMOVE DC COMPONENT -- E.G., CHEBYCHEV
dpSpectrum(:,1:16) = 0;

%% low pass filter
expFilter = exp((-(0:length(dpSpectrum)-1))*loPass);
dpSpectrum = dpSpectrum.*expFilter;

%% Return function variable: matrix of distorted signals in frequency domain
A = dpSpectrum;

%% Graphing and other stuff
if doPlot == 1
    plotting = abs(dpSpectrum);
    bins = 0:lenPosSpec-1;
    
    figure(1)
    stairs(bins, spectra(1,1:lenPosSpec))
    axis([0 2000 0 1.1*max([plotting spectra(1,1:lenPosSpec)])])
    
    figure(2)
    stairs(bins, plotting)
    axis([0 2000 0 1.1*max([plotting spectra(1,1:lenPosSpec)])])
end

end