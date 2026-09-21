function names = xmlColumns()
%XMLCOLUMNS  The columns internal.parseMusicXml produces, in order.
%
%   Twin of the Python mpt.score._XML_COLUMNS.
    names = {'onsetBeats', 'onsetSeconds', 'durationBeats', ...
             'durationSeconds', 'pitch', 'velocity', 'part', 'voice', ...
             'measure', 'fermata'};
end
