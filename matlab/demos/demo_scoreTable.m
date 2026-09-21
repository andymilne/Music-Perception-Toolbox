%% demo_scoreTable.m — the event table returned by readScore
%
%  readScore returns a MATLAB table, one row per sounding note, so a score
%  is inspected and filtered with MATLAB itself and no toolbox function is
%  needed to read it. preMaetFromScore accepts either the path or the
%  table, so this reader is called directly only when the table is wanted.
%
%  The Python sibling is demos/demo_score_table.py.
%
%  See also READSCORE, PREMAETFROMSCORE.

score = fullfile(fileparts(mfilename('fullpath')), 'jmm', 'data', ...
                 'bwv347.musicxml');
t = readScore(score);

%% What comes back
fprintf('%d notes from a %s score\n\n', height(t), t.Properties.Description);
disp(head(t, 4));

% part is categorical and its categories are the part names, so the names
% are in the column rather than in a field beside it.
fprintf('parts: %s\n', strjoin(categories(t.part).', ', '));

% A MusicXML score carries voice and fermata; a MIDI file would carry
% channel instead, and neither stands in for the other.
fprintf('columns this source carries: %s\n\n', ...
        strjoin(t.Properties.VariableNames, ', '));

%% Selection is MATLAB
soprano = t(t.part == 'Soprano', :);
fprintf('soprano notes: %d, range %g-%g\n', height(soprano), ...
        min(soprano.pitch), max(soprano.pitch));

held = t(t.fermata, :);
fprintf('notes under a fermata: %d, at beats %s\n\n', height(held), ...
        mat2str(unique(held.onsetBeats).'));

% Anything a table does, this table does: here, the mean duration of each
% part, which no toolbox function has to provide.
disp('mean duration in quarter notes, by part:');
disp(groupsummary(t, 'part', 'mean', 'durationBeats'));

%% Sampling on a grid
% Analysis 1.1 of the JMM article reads this chorale on a sixteenth-note
% grid. gridEvents is table to table, so the result is selected from the
% same way; a held note replicates across the points it covers, and a
% point with nothing sounding would be a row whose note columns are all
% missing (this chorale has no rests).
g = gridEvents(t, 0.25);
disp('--- on a sixteenth-note grid ---');
fprintf('grid points: %d, rows: %d, empty points: %d\n', ...
        numel(unique(g.gridIndex)), height(g), sum(isnan(g.noteId)));
disp('the first two points:');
disp(head(g(:, {'gridIndex', 'gridOnsetBeats', 'pitch', 'part', 'weight'}), 8));

%% On to a pre-MAET
% The table goes to preMaetFromScore in place of the path, so any
% selection made above carries through.
pm = preMaetFromScore(soprano, 'attributes', {'pitch', 'onset'}, ...
                      'time', 'beats', 'chords', 'separate');
[p, ~, specs] = unpackPreMaet(pm);
fprintf('pre-MAET from the soprano alone: %d events, attributes %s\n', ...
        size(p{1}, 2), strjoin(cellfun(@(s) s.name, specs, ...
                                       'UniformOutput', false), ', '));
