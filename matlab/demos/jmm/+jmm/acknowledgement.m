function t = acknowledgement(midiPath)
%ACKNOWLEDGEMENT  The attribute table of Coltrane's Acknowledgement (solo).
%
%   t = jmm.acknowledgement()
%   t = jmm.acknowledgement(midiPath)
%
%   Reads a monophonic MIDI transcription with readScore --- by default
%   data/AwakeningSolo.mid --- and returns the attribute table, one row per
%   note, in onset order (onsetBeats, durationBeats, pitch, ...).
%
%   The transcription is not part of the toolbox distribution: place
%   your own transcription of the solo at that path, or pass its path.
%
%   See also READSCORE, JMM.BWV347NOTES.
    if nargin < 1 || isempty(midiPath)
        midiPath = fullfile(jmm.dataDir(), 'AwakeningSolo.mid');
    end
    if ~isfile(midiPath)
        error('jmm:acknowledgementMissing', ...
            ['%s not found. The transcription of Acknowledgement is not ' ...
             'distributed with the toolbox; place a monophonic MIDI ' ...
             'transcription of the solo at that path (or pass its path).'], ...
            midiPath);
    end
    t = sortrows(readScore(midiPath), 'onsetBeats');
end
