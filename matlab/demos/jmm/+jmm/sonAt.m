function son = sonAt(t)
%SONAT  SATB sonority (4 MIDI pitches) sounding at QN time t.
%
%   son = jmm.sonAt(t)
%
%   Twin of bwv_window.son_at in the Python demos.
%
%   See also JMM.BWVWINDOWSTATE, JMM.ISROOTPOSITION, JMM.ISSIXFOUR.
    S = jmm.bwvWindowState();
    son = S.satb(round((t - S.T0) / S.gridStep) + 1, :);
end
