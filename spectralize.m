function [S, Partials_p, Partials_w] = spectralize(x_p, x_w, spec_p, spec_w)
%SPECTRALIZE Add weighted partials to weighted pitches
%
%   [S, partials_p, partials_w] = spectralize(x_p, x_w, spec_p, spec_w)
%
%   Given a pitch multiset with pitches x_p and corresponding weights x_w, 
%   and a multiset of spectral pitches and weights (spec_p and spec_w) to be 
%   added to each element of the pitch multiset,
%
%   S = spectralize(x_pc, x_w, spec_p, spec_w)
%
%   returns a matrix S of all spectral pitches and weights. The first row
%   comprises the pitches of all partials, the second row comprises their
%   weights. This function optionally returns the length(specPc) by
%   length(scalePc) matrices of spectral pitches and weights, where the nth
%   column is the spectrum for x_p[n].
%
%   If x_w is 0 or empty all pitches in the multiset have unit weight; likewise
%   with y_w.


% Make all inputs column vectors
x_p = x_p(:);
if isempty(x_w) || isequal(x_w,0)
    x_w = ones(numel(x_p),1);
end
x_w = x_w(:);
spec_p = spec_p(:);
spec_w = spec_w(:);

% Calculate all pitch classes
Partials_p = spec_p' + x_p;

% Calculate all weights
Partials_w = spec_w'.*x_w;

S = [Partials_p(:) Partials_w(:)];

end