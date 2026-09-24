function spans = bwv347FermataSpans()
%BWV347FERMATASPANS  Quarter-note spans of the fermata-bearing notes.
%
%   spans = jmm.bwv347FermataSpans()
%
%   Returns an M x 2 matrix of [start, end] quarter-note spans of the
%   fermata-bearing notes of the played-through chorale, sorted and
%   without duplicates, from the .fermata column of the attribute table
%   (readScore sets it to 1 where a MusicXML note carries a fermata).
%   Analysis 1.3 raises the weight of every eighth-note event under a
%   fermata by half.
%
%   See also JMM.BWV347NOTES, JMM.BWVWINDOWSTATE.
    t = jmm.bwv347Notes();
    f = t.fermata;
    spans = unique([t.onsetBeats(f), t.onsetBeats(f) + t.durationBeats(f)], ...
                   'rows');
end
