function b = b2bar(t)
%B2BAR  Played-through bar coordinate from QN time (pickup at 0--1).
%
%   b = jmm.b2bar(t)
%
%   b = (t - 1) / 4 + 1, elementwise.
%
%   See also JMM.BWV347BAR.
    b = (t - 1.0) / 4.0 + 1.0;
end
