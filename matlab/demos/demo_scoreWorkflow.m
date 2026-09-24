%% demo_scoreWorkflow.m — from a score (musicXML, MIDI) to a MAET analysis
%
%  This demo is the spine of the demo_score* family. It reads a score into
%  an attribute table, looks at that table, samples it on a grid, encodes a
%  categorical column, builds the density, and runs an analysis on it.
%  Where a step has more to it than the one choice made here, a comment
%  names the demo that goes further:
%
%    demo_scoreGrid          choosing the grid step and the weighting
%    demo_scoreCategoricals  the three ways to encode a category
%
%  The functions that get a score to a pre-MAET:
%
%    score file (MIDI or MusicXML)
%      readScore              -> attribute table, sampled per note
%    attribute table (attributes across the columns)
%      gridAttrTable          -> attribute table, sampled on a time grid
%      ungridAttrTable        -> attribute table, the time grid undone
%      preMaetFromAttrTable   -> pre-MAET
%    pre-MAET (rows are attributes, columns are events; the values
%    (pAttr), the weights (wAttr), and the per-attribute parameters
%    (specs) are its three parts, and showPreMaet renders the specs as a
%    leading column)
%      unpackPreMaet          -> its three parts (pAttr, wAttr, specs)
%      packPreMaet            -> a pre-MAET from those parts
%      showPreMaet               display it
%      writePreMaet              write it as markdown, LaTeX, or CSV
%      readPreMaet            -> a pre-MAET read back from that, which
%                                stands in for every step above
%
%  Rows of attribute tables are selected with MATLAB's own indexing rather
%  than a toolbox function. What happens to a pre-MAET next --
%  selectPreMaet and the other preprocessing operations, buildMaet, and the
%  measures -- is a separate family, and demo_preprocessing covers it.
%
%  The conversion from attribute table to pre-MAET transposes, and
%  regroups. An attribute table carries its attributes across the columns,
%  as any data table does; a pre-MAET carries attributes down the rows and
%  events across the columns, which is the layout of the article's pre-MAET
%  table and of showPreMaet's output, so a table's column becomes a
%  pre-MAET's row and each attribute's values are a K_a x N matrix. This
%  horizontal/wide format is preferred because it corresponds to that used
%  in musical scores and DAWs.
%
%  The events are not the attribute table's rows. The conversion to
%  pre-MAET gathers rows into events -- a chord bound into one event, or a
%  grid point holding its voices -- so N counts events and K_a counts the
%  values one event carries on that attribute. Here 1088 attribute table
%  rows become 272 pre-MAET events of four pitches each.
%
%  Only three stages are needed to get from a score (MIDI or MusicXML) to a
%  MAET density: readScore, preMaetFromAttrTable, buildMaet. Everything
%  between them is optional. This demo takes readScore, gridAttrTable,
%  preMaetFromAttrTable, selectPreMaet, buildMaet, simMaet: it grids
%  because the structural role of step 4 needs every event to hold the same
%  voices, and it selects because the question is about two chords out of
%  the 272.
%
%  See also READSCORE, GRIDATTRTABLE, PREMAETFROMATTRTABLE, SELECTPREMAET.

% The chorale ships with the demos, and is located from the toolbox root.
mptRoot = which('buildMaet');
assert(~isempty(mptRoot), 'demoScore:toolboxNotFound', ...
       'Add the toolbox''s matlab folder to the path, then run again.');
score = fullfile(fileparts(mptRoot), 'demos', 'jmm', 'data', ...
                 'bwv347.musicxml');
clear mptRoot

%% 1. Read
% One row per sounding note. A column is present only where the source
% carries it: this is MusicXML, so it has voice, staff, fermata, and the
% articulations. A MIDI file would instead have channel, program,
% noteNumber, weight, and soundingDuration (the last two being the
% loudness controllers and the pedals resolved into the note's own
% columns). See readScore's docstring.
t = readScore(score);
fprintf('%d notes from a %s score; the first five rows:\n', ...
        height(t), t.Properties.Description);
disp(head(t, 5));

%% 2. Look and select
% The return is a MATLAB table, inspected and filtered with MATLAB.
fprintf('parts: %s\n', strjoin(categories(t.part).', ', '));
fprintf('pitch range: %g to %g\n', min(t.pitch), max(t.pitch));
fprintf('notes under a fermata: %d\n', sum(t.fermata));

% A selection carries through everything below, and is written as a
% MATLAB row selection. Here we keep the whole chorale.
fprintf('the first five notes of t(t.part ~= ''Bass'', :):\n');
disp(head(t(t.part ~= 'Bass', :), 5));

%% 3. Sample on a grid
% A grid makes the event index a uniform index of time, and gives every
% event the same voices, which the structural encoding of step 4 needs.
% A sixteenth is the shortest note value in this example.
% See demo_scoreGrid for the step and the weight policies.
g = gridAttrTable(t, 0.25);
fprintf('gridded: %d points, %d rows; the first five:\n', ...
        numel(unique(g.gridIndex)), height(g));
disp(head(g(:, {'gridIndex', 'gridOnsetBeats', 'noteId', 'weight', ...
                'pitch', 'part', 'durationBeats'}), 5));

%% 4. Convert to a pre-MAET
% An attribute is a column of the attribute table read under a set of
% parameters (specs), so each entry names both. Three things worth knowing
% happen here.
%
% 'pitch' is here listed twice, so the same values become two attributes
% under different specs: one reads pitch class, wrapping at the octave,
% and the other pitch height, which does not. Their product is Shepard's
% helix, and the helix's pitch height axis can be stretched or compressed 
% by changing its sigma value.
%
% A score fixes what the values are and not how tolerant a match is, nor
% how many of an event's values a tuple takes, so sigma, r, and exch are
% the analyst's and the conversion asks for them. It fills in only what
% follows from the data or from another argument here: r and exch under a
% structural role, and 'read as written' for rel and isPer.
%
% An attribute table with a categorical column needs to be assigned a role,
% which determines how its levels are represented in the pre-MAET. In this
% example, part (Soprano, Alto, Tenor, Bass) is given a role. There are
% three roles:
%   'orderedMultiset'    structural: the level becomes a position within
%                        one attribute, so the four voices occupy four
%                        slots and matching is voice by voice. Used here,
%                        and it fixes r = 4 and exch = false, which is why
%                        neither is given.
%   'separateAttributes' structural: the level becomes an attribute of
%                        its own, one per voice.
%   'simplex'            a value: the level becomes the coordinates of a
%                        simplex vertex on an attribute of its own, so
%                        two chords can match on some voices and not
%                        others.
% See demo_scoreCategoricals for how these differ when applied to the same 
% music.
%
% The name-value pairs after the attributes carry the rest of the
% reading. 'time', 'beats' puts the onset attribute's values in quarter
% notes rather than in the default seconds, matching the unit the grid of
% step 3 was built over.
pm = preMaetFromAttrTable(g, 'attributes', { ...
        struct('column', 'pitch', 'name', 'pitchClass', 'sigma', 0.5, ...
               'isPer', true, 'period', 12), ...
        struct('column', 'pitch', 'name', 'pitchHeight', 'sigma', 8), ...
        struct('column', 'onset', 'sigma', 0.5)}, ...
        'time', 'beats', 'roles', struct('part', 'orderedMultiset'));
[pAttr, wAttr, specs] = unpackPreMaet(pm);
showPreMaet(pm, 'maxEvents', 4, 'title', 'the pre-MAET, first events');
fprintf('\n');

%% 5. Select what the question is about
% selectPreMaet keeps a selection of a pre-MAET's attributes and of its
% events, in the order given, and returns a pre-MAET like any other. The
% question below is about two chords, compared on their pitches, so each
% chord becomes a one-event pre-MAET on the two pitch attributes; onset
% located them and is not compared on. The four values are in S, A, T, B
% order, the slots the orderedMultiset role gave them.
%
% The two chords are the final chords of the first two cadences, which
% fall on beats 7 and 15. The onset attribute is searched for those two
% beats to get their event indices; on this grid of sixteenths they are
% not events 7 and 15.
pmFull = packPreMaet(pAttr, wAttr, specs);
onsets = pAttr{3}(1, :);
events = [find(abs(onsets - 7) < 1e-9, 1), find(abs(onsets - 15) < 1e-9, 1)];
cadence = cell(1, 2);
for k = 1:2
    cadence{k} = selectPreMaet(pmFull, ...
        'attributes', {'pitchClass', 'pitchHeight'}, 'events', events(k));
    showPreMaet(cadence{k}, 'title', sprintf('cadence %d tonic', k));
    fprintf('\n');
end

%% 6. Run a MAET analysis
% The two tonic chords are the same four pitch classes, differing only in
% the octave of the bass. Whether that counts as the same chord is what the
% pitch-height sigma decides. buildMaet's 'sigma' overrides the specs', so
% the sweep over pitch height's sigma needs no rebuild.
disp('similarity of the two, against the pitch-height width:');
for sigmaHeight = [1 4 16 64]
    dens = cell(1, 2);
    for k = 1:2
        dens{k} = buildMaet(cadence{k}, 'sigma', [0.5 sigmaHeight], ...
                            'verbose', false);
    end
    fprintf('  sigma_pitchHeight = %5.1f semitones   ->  %.3f\n', ...
            sigmaHeight, simMaet(dens{1}, dens{2}, 'verbose', false));
end
fprintf(['\nnarrow: two different chords, the bass octave counting.\n' ...
         'wide:   one chord, the octave forgiven and the pitch classes ' ...
         'agreeing.\n']);
