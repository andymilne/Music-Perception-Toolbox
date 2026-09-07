function tf = isSixFour(son)
%ISSIXFOUR  Pitch-derived second-inversion test: a perfect fourth above the bass.
%
%   tf = jmm.isSixFour(son)
%
%   Twin of bwv_window.is_six_four in the Python demos.
%
%   See also JMM.ISROOTPOSITION, JMM.SONAT.
    tf = any(jmm.pcsAboveBass(son) == 5);
end
