function [v,vCentred] = circConv(signal,kernel)
%CIRCCONV Circular convolution modulo the dimension of longest vector.
%   [v vCentred] = CIRCCONV(A, B) circularly convolves vectors A and B.
%
%   See also: gaussianKernel

if any(isnan(signal))
    error('All NaNs must be removed from the signal vector.')
end

N = length(signal);
M = length(kernel);
if M == 1
    v = kernel*signal;
    vCentred = v;
else
    v = ifft(fft(signal(:),N).*fft(kernel(:),N),N);
    vCentred = circshift(v,-ceil(M/2));
end

end