function b = b2bar(t)
%B2BAR  Played-through bar coordinate from QN time (pickup at 0--1).
%
%   b = jmm.b2bar(t)
%
%   b = (t - 1) / 4 + 1, elementwise.
%
%   Twin of bwv_window.b2bar in the Python demos.
%
%   See also JMM.BWV347BAR.
    b = (t - 1.0) / 4.0 + 1.0;
end
