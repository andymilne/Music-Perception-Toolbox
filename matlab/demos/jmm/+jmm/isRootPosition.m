function tf = isRootPosition(son)
%ISROOTPOSITION  Pitch-derived root-position test.
%
%   tf = jmm.isRootPosition(son)
%
%   Per the specified rules: a fifth above the bass; or a major or minor
%   third above the bass with no fourth, no fifth, and no sixth. son is a
%   vector of MIDI pitches (NaN entries ignored).
%
%   Twin of bwv_window.is_root_position in the Python demos.
%
%   See also JMM.ISSIXFOUR, JMM.SONAT.
    pcs = jmm.pcsAboveBass(son);
    if any(pcs == 7)
        tf = true;
        return;
    end
    tf = any(ismember([3 4], pcs)) && ~any(ismember([5 8 9], pcs));
end
