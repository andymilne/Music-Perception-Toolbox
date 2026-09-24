function tf = isSixFour(son)
%ISSIXFOUR  Pitch-derived second-inversion test: a perfect fourth above the bass.
%
%   tf = jmm.isSixFour(son)
%
%   See also JMM.ISROOTPOSITION, JMM.SONAT.
    tf = any(jmm.pcsAboveBass(son) == 5);
end
