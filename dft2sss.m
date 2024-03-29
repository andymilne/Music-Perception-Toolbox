function X = dft2sss(dft,Fs,scaleTypes,varargin)

%dft2sss Converts the vector returned by FFT into a sparse sound structure.
%
%   Authors: Andrew J. Milne
%   Revision:
%   Date: 2017/11/08
%
p = inputParser;
addParameter(p,'FRef',261.6256); %% middle C
parse(p,varargin{:});
fRef = p.Results.FRef;

if size(dft,2) > 1
    error('dft must be column vector; e.g., from a mono audio signal')
end

nSamples = length(dft);
binWidth = Fs/nSamples; % Hertz

frequencies = (0 : binWidth : Fs/2)';
phasors = dft(1:length(frequencies));

X = struct('F',frequencies,'Phasors',phasors);

% Musical pitch (cents)
if any(contains(scaleTypes,'MP','IgnoreCase',true))
    X.MP = 1200*(log2(X.F) - log2(fRef)); 
end
% test = [X.F X.MP abs(X.Phasors)]
% test = test(1:1000,:)

% Equisection pitch (mels)
if any(contains(scaleTypes,'ESP','IgnoreCase',true))
    X.ESP = X.F./(0.759 + 0.000252.*X.F); 
end

% Critical band rate (Traunmüller, 1990) (barks)
if any(contains(scaleTypes,'CBR','IgnoreCase',true))
    X.CBR = 26.81*X.F./(1960 + X.F) - 0.53; 
end

% Equivalent rectangular bandwidth rate (Moore & Glasberg, 1983) (ERB units or Cams)
if any(contains(scaleTypes,'ERBR','IgnoreCase',true))
    X.ERBR = 11.17*(log(X.F + 312) - log(X.F + 14675)) + 43; 
end

% Inverse Greenwood function (Greenwood function)
if any(contains(scaleTypes,'IGF','IgnoreCase',true))
    X.IGF = 0.47619*log10(X.F/165.4 + 0.88); 
end

end
