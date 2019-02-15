function gKer = gaussianKernel(sigma, winLen)

%% Truncated gaussian kernel of std dev sigma and length winLen SDs.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University
%
%   See also: pSetSpectralEntropy

%% Build smoothing kernel
windowLength = ceil(winLen*sigma);
if windowLength/2 == floor(windowLength/2)
    windowLength = windowLength + 1;
end
n = -(windowLength - 1)/2 : (windowLength - 1)/2;
gKer = exp(-(n.^2)/(2*sigma^2));

end

