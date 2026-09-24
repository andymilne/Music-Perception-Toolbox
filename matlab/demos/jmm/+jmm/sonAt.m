function son = sonAt(t)
%SONAT  The sonority (MIDI pitches) sounding at QN time t.
%
%   son = jmm.sonAt(t)
%
%   See also JMM.BWVWINDOWSTATE, JMM.ISROOTPOSITION, JMM.ISSIXFOUR.
    S = jmm.bwvWindowState();
    rows = abs(S.sixteenths.gridOnsetBeats - t) < 1e-9;
    son = double(S.sixteenths.pitch(rows)).';
end
