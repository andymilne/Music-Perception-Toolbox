function [edge_w, hiResEdge_w] = edges(x_ind,mu,kappa)
%EDGES Edge detection using derivative of von Mises kernel.
%   Given a weighted indicator vector x_ind, which represents periodic
%   notes/onsets by nonnegative weights and pitch/time by index, edge_w
%   gives the edge weights ("edginess") of each pitch/time class.
%
%   This function adapts the method used in image processing to detect
%   edges by convolving with the first derivative of a Gaussian kernel and
%   taking the absolute values. Because x_ind is 1-dimensional and
%   circular, the first derivative of a von Mises kernel is used (with
%   parameters mu and kappa), and this is circularly convolved with the
%   indicator vector.
%
%   [edge_w, hiResEdge_w] = edges(x_ind,mu,kappa) also returns the
%   hi-resolution and signed edge weights.
%
%   By Andrew J. Milne, The MARCS Institute, Western Sydney University

if nargin < 3
    kappa = 6.7;
end
if nargin < 2
    mu = 0;
end

N = numel(x_ind);
eventIndex = find(x_ind);
K = numel(eventIndex);

hirez_pattern = zeros(1,N*100);
hirez_pattern(100*(eventIndex-1) + 1) = 1;

x = -pi:2*pi/(N*100):pi - 2*pi/N;

% Create von Mises kernel to find offset so it is centered around mu
vM_win = exp(kappa.*cos(x-mu)) / (2*pi * besseli(0,kappa));
[~, vM_win_center] = max(vM_win);

% Create first derivative of von Mises kernel and convolve with pattern.
% Offset resulting pattern so it is centered around mu
vM_dx_win_hirez ...
    = - kappa.*sin(x-mu) ...
    .* exp(kappa.*cos(x-mu)) ...
    / (2*pi * besseli(0,kappa));
vM_con_pattern_hirez ...
    = cconv(hirez_pattern,vM_dx_win_hirez,length(hirez_pattern));
hiResEdge_w = circshift(vM_con_pattern_hirez,[0 -vM_win_center]);

% Sample at pulse/chromatic level
edge_w = abs(hiResEdge_w(1:100:N*100));

end