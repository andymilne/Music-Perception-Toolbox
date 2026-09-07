function b = bwv347Bar(t)
%BWV347BAR  Played-through bar of a grid time in quarter notes.
%
%   b = jmm.bwv347Bar(t)
%
%   The chorale has a one-quarter pickup at 0--1 QN, then 4/4 bars from
%   1 QN: bar 0 for t < 1, otherwise floor((t - 1) / 4) + 1. Elementwise
%   over an array of times.
%
%   Twin of jmm_data.bwv347_bar in the Python demos.
    b = zeros(size(t));
    m = t >= 1.0;
    b(m) = floor((t(m) - 1.0) / 4.0) + 1;
end
