function t = bwv347Notes()
%BWV347NOTES  The attribute table of the played-through chorale (readScore).
%
%   t = jmm.bwv347Notes()
%
%   Reads data/bwv347.musicxml with readScore --- Bach, "Ich dank dir,
%   lieber Herre" (BWV 347), with the bars 1--4 repeat expanded --- and
%   returns the attribute table (a MATLAB table with one row per note:
%   onsetBeats, durationBeats, pitch, part, voice, fermata, ...). The
%   article used the same score from the music21 corpus; the two
%   encodings agree to the note. The table is read once and cached in a
%   persistent variable.
%
%   See also READSCORE, JMM.BWV347FERMATASPANS, JMM.ACKNOWLEDGEMENT.
    persistent cached
    if isempty(cached)
        cached = readScore(fullfile(jmm.dataDir(), 'bwv347.musicxml'));
    end
    t = cached;
end
