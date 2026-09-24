%% demo_scoreGrid.m — sampling an attribute table on a time grid
%
%  gridAttrTable samples an attribute table on a time grid, and this demo
%  works through the three choices it takes: the step, what a slice takes
%  from a note that overlaps it, and what to do with a slice where
%  nothing sounds. demo_scoreWorkflow makes one of each in passing -- a
%  sixteenth step, the default weighting -- and points here.
%
%  See also GRIDATTRTABLE, READSCORE, PREMAETFROMATTRTABLE.

% The chorale ships with the demos, and is located from the toolbox root.
mptRoot = which('buildMaet');
assert(~isempty(mptRoot), 'demoScore:toolboxNotFound', ...
       'Add the toolbox''s matlab folder to the path, then run again.');
score = fullfile(fileparts(mptRoot), 'demos', 'jmm', 'data', ...
                 'bwv347.musicxml');
clear mptRoot

t = readScore(score);
fprintf('%d notes over %g beats\n\n', height(t), ...
        max(t.onsetBeats + t.durationBeats));

% A held note: the tenor's B, entering on beat 5 and lasting a beat and a
% half, so that on a beat grid it fills one slice and half of the next.
held = 26;

% One pitch-class attribute, for the identities below: every note's pitch
% read as a class, singly.
pitchClass = struct('column', 'pitch', 'name', 'pitchClass', ...
                    'sigma', 0.5, 'isPer', true, 'period', 12);

% The columns printed below, a readable subset of the gridded table's.
cols = {'gridIndex', 'gridOnsetBeats', 'noteId', 'weight', 'pitch', ...
        'part', 'durationBeats'};

%% 1. The step
% A slice is a step long, and a note belongs to every slice it overlaps,
% so nothing falls between samples however fine or coarse the grid. The
% step therefore sets how much of the score each event gathers, not
% whether an event is seen.
disp('step   points   rows');
for step = [0.25 0.5 1 4]
    g = gridAttrTable(t, step);
    fprintf('%5g   %6d   %4d\n', step, numel(unique(g.gridIndex)), height(g));
end
gBeat = gridAttrTable(t, 1);
fprintf('\nthe first six rows at a step of one beat:\n');
disp(gBeat(1:6, cols));
fprintf(['\nA step at or below the shortest note value gives one row per ' ...
         'note, and\nthe grid is then a re-indexing of the score by time. ' ...
         'A coarser step\ngathers several notes of a voice into one event, ' ...
         'and the\nweighting below then applies.\n\n']);

%% 2. The weighting
% What a slice takes from a note that overlaps it. On a beat grid the
% three policies differ, because notes and slices no longer coincide.
disp('the tenor''s dotted-quarter B across the two slices it occupies:');
policies = {'coverage', 'presence', 'item'};
for k = 1:3
    g = gridAttrTable(t, 1, 'weights', policies{k});
    rows = g(g.noteId == held, :);
    fprintf('  %-9s', policies{k});
    for r = 1:height(rows)
        fprintf('  slice %d: %.3f', rows.gridIndex(r), rows.weight(r));
    end
    fprintf('\n');
end
fprintf('\nits two rows under ''coverage'':\n');
disp(gBeat(gBeat.noteId == held, cols));
fprintf([ ...
    '\n  coverage  the fraction of the slice the note fills: how the ' ...
    'span is\n            filled. It is full in the slice it covers and ' ...
    'half in the\n            slice it half covers.\n' ...
    '  presence  full weight wherever the note appears at all: which ' ...
    'notes\n            are here, not how much of the slice each holds.\n' ...
    '  item      the fraction of the note in the slice, so that a note\n' ...
    '            counts once however many slices it spans.\n\n']);

% Two of the three are re-weightings of the ungridded attribute table: on an attribute
% constant over the note and read at r = 1, each gives back a density the
% ungridded table already had.
disp('against the ungridded score, at r = 1:');
pairs = {'coverage', 'duration'; 'item', 'ones'};
for k = 1:size(pairs, 1)
    g = gridAttrTable(t, 1, 'weights', pairs{k, 1});
    dGrid = buildMaet(preMaetFromAttrTable(g, 'attributes', {pitchClass}, ...
        'time', 'beats', 'chords', 'separate', 'weights', 'weight'), ...
        'verbose', false);
    dNotes = buildMaet(preMaetFromAttrTable(t, 'attributes', {pitchClass}, ...
        'time', 'beats', 'chords', 'separate', 'weights', pairs{k, 2}), ...
        'verbose', false);
    fprintf('  %-9s grid  ~  %-8s notes   %.3f\n', pairs{k, 1}, ...
            pairs{k, 2}, simMaet(dGrid, dNotes, 'verbose', false));
end
fprintf(['\nso at r = 1 a grid is a re-weighting. It applies beyond ' ...
         'that where the\nevents have to line up: a structural role ' ...
         'needs every event to hold the\nsame voices ' ...
         '(-> demo_scoreCategoricals), and differencing needs the ' ...
         'event\nindex to be a uniform index of time.\n\n']);

%% 3. Empty slices
% A slice with nothing sounding is kept as one row whose note columns are
% all missing. It holds the place that makes the event index uniform, and
% downstream it is an event contributing no tuple while keeping its
% position. Selecting the fermata notes and gridding the selection makes
% plenty of them.
fermatas = t(t.fermata, :);
g = gridAttrTable(fermatas, 1);
empty = sum(isnan(g.noteId));
fprintf('%d fermata notes over %d slices: %d rows, of which %d are empty\n', ...
        height(fermatas), numel(unique(g.gridIndex)), height(g), empty);
disp(g(5:10, cols));
fprintf('dropping them is one selection away: %d rows\n\n', ...
        height(g(~isnan(g.noteId), :)));

%% 4. Further points
% 'limits' fixes the span the grid covers, which is how two pieces are
% put on the same grid. The default runs from 0 to the last note's end.
gEight = gridAttrTable(t, 1, 'limits', [0 8]);
fprintf('first 8 beats only: %d slices\n', numel(unique(gEight.gridIndex)));
disp(gEight(end-3:end, cols));

% 'duration' chooses which duration defines occupancy. A score has only
% the notated one; a MIDI file also has soundingDuration, the notated one
% with the sustain and sostenuto pedals resolved into it, and gridding
% over that is what makes a pedalled performance read as held rather than
% detached.
fprintf(['this table has a sounding duration: %d — it is a score, and a ' ...
         'score carries no pedal\n'], ...
        any(startsWith(t.Properties.VariableNames, 'soundingDuration')));

% The grid steps in one unit but its points have a time in both, so a
% metrical grid can be read on a clock: slices of a sixteenth, and a
% sigma in milliseconds. The unit the grid did not step in is
% interpolated from the attribute table's note samples, so it is exact wherever the
% tempo is constant and approximate only across a tempo change.
gBeat = gridAttrTable(t, 0.25);
fprintf('a beat grid''s first four points in each unit:\n');
disp(head(unique(gBeat(:, {'gridOnsetBeats', 'gridOnsetSeconds'}), ...
                 'rows', 'stable'), 4));

% ungridAttrTable is the inverse: noteId says which row of the source
% each grid row came from, so keeping the first of each and removing what
% the grid wrote returns the attribute table it came from. Which columns those are
% is not a fixed list -- the grid adds weight to an attribute table that had none
% and overwrites the weight of one that did -- which is why this is a
% toolbox function and not four lines of MATLAB in the caller.
back = ungridAttrTable(gBeat);
fprintf('\nungridded: %d rows, against the %d read; identical: %d\n', ...
        height(back), height(t), isequal(back.pitch, t.pitch));
