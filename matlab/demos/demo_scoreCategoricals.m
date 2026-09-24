%% demo_scoreCategoricals.m — encoding a categorical column: three ways
%
%  A categorical column reaches a pre-MAET by one of three roles, and
%  this demo contrasts them on one chorale: what question each asks, and
%  how they are related. Voice is the example, but the same three carry
%  any category an attribute table holds -- instrument, articulation, an
%  experimental condition, a cluster label. demo_scoreWorkflow makes one
%  of these choices in passing and points here.
%
%  See also PREMAETFROMATTRTABLE, SELECTPREMAET, GRIDATTRTABLE.

% The chorale ships with the demos, and is located from the toolbox root.
mptRoot = which('buildMaet');
assert(~isempty(mptRoot), 'demoScore:toolboxNotFound', ...
       'Add the toolbox''s matlab folder to the path, then run again.');
score = fullfile(fileparts(mptRoot), 'demos', 'jmm', 'data', ...
                 'bwv347.musicxml');
clear mptRoot

g = gridAttrTable(readScore(score), 0.25);

% The attributes every encoding below converts to. Pitch class wraps at
% the octave and is read strictly; pitch height is read loosely, so that a
% displacement of register is forgiven and what is left to disagree about
% is the voicing. demo_scoreWorkflow sweeps that width.
attributes = {struct('column', 'pitch', 'name', 'pitchClass', ...
                     'sigma', 0.5, 'isPer', true, 'period', 12), ...
              struct('column', 'pitch', 'name', 'pitchHeight', 'sigma', 24), ...
              struct('column', 'onset', 'sigma', 0.5)};

% The simplex role builds an attribute of its own, so it carries its own
% width rather than taking one from the list above.
voice = struct('role', 'simplex', 'sigma', 0.2);

%% The three encodings
% The two structural roles realize the level as *where the value sits*:
% 'separateAttributes' gives each level an attribute of its own, and
% 'orderedMultiset' gives each level a fixed position within one attribute.
% Either way, the binding of a value to its level is carried by the layout,
% and an event holds the whole chord. What is adjustable afterwards differs:
% separate attributes carry their own kernel parameters and can be selected
% one at a time, but each holds a single value, so r = 1 and every level is
% always read together; one ordered multiset shares a kernel across the
% levels and takes r > 1, which is how some of the voices are read at a
% time rather than all of them. 'orderedMultiset' is the one used here.
aware = localConvert(g, attributes, 'roles', struct('part', 'orderedMultiset'));

% The value role, 'simplex', realizes the level as *a value of its own*, on
% its own attribute, so each note becomes its own event and the binding is
% the product of the two attributes at that event.
simplex = localConvert(g, attributes, 'chords', 'separate', ...
                       'roles', struct('part', voice));

% No role at all, at the same one-event-per-note grain: the voice is not
% part of the encoding.
agnostic = localConvert(g, attributes, 'chords', 'separate');

names = {'voice-aware', 'simplex-voice', 'voice-agnostic'};
pms   = {aware, simplex, agnostic};
for k = 1:3
    [pAttr, ~, specs] = unpackPreMaet(pms{k});
    fprintf('%-15s %d attributes, %4d events, r = %s\n', names{k}, ...
            numel(pAttr), size(pAttr{1}, 2), ...
            mat2str(cellfun(@(s) s.r, specs)));
end
fprintf('\n');
for k = 1:3
    showPreMaet(pms{k}, 'maxEvents', 4, 'title', names{k});
    fprintf('\n');
end

%% What each asks
% Two E major chords a beat apart, in quite different voicings:
% (64, 59, 56, 40) and (71, 68, 64, 52). The same four pitch classes in
% both, but only the bass keeps its own, and every voice moves in
% register.
disp('similarity of two voicings of one chord:');
perChord = [1, 4, 4];
for k = 1:3
    % Compare on the pitch content and the voice encoding, not on when
    % the chord happens: the onset attribute is what located it.
    [~, ~, specs] = unpackPreMaet(pms{k});
    allNames = cellfun(@(s) s.name, specs, 'UniformOutput', false);
    keep = allNames(~strcmp(allNames, 'onset'));
    beats = [7, 8];
    dens = cell(1, 2);
    for c = 1:2
        dens{c} = buildMaet(selectPreMaet(pms{k}, ...
            'attributes', keep, ...
            'events', localChordEvents(pms{k}, beats(c), perChord(k))), ...
            'verbose', false);
    end
    fprintf('  %-15s %.3f\n', names{k}, ...
            simMaet(dens{1}, dens{2}, 'verbose', false));
end

fprintf([ ...
    '\n  voice-aware    asks whether *all* voices match, so the ' ...
    're-voicing\n                 zeroes the product: a multiplicative ' ...
    'AND across voices,\n                 and a pitch-class mismatch ' ...
    'never relaxes with width.\n' ...
    '  simplex-voice  asks what *fraction* of voices match: the bass ' ...
    'alone\n                 keeps its pitch class, and contributes its ' ...
    'quarter.\n' ...
    '  voice-agnostic asks whether the *pitch contents* match, and they ' ...
    'do —\n                 the same multiset of pitch classes, voiced ' ...
    'differently.\n\n']);

%% The family relation
% The agnostic encoding is the simplex one with its voice attribute
% removed, which is one selection rather than another conversion.
% The attributes are selected by index rather than by name, because the
% two pitch attributes share a name and a name selects the first of them.
withoutVoice = selectPreMaet(simplex, 'attributes', [1 2 3]);
same = isequal(unpackPreMaet(withoutVoice), unpackPreMaet(agnostic));
fprintf('dropping the voice attribute recovers the agnostic reading: %d\n\n', same);

%% One trap
% With bound chords and no role, nothing has fixed exch, so the
% conversion asks; exch = true reads the chord as an unordered multiset on
% *every* attribute, and two attributes describing the same notes then
% pair every value of one with every value of the other — the soprano's
% pitch class with the bass's height. Binding a note's attributes to each
% other needs one event per note, which is what the agnostic encoding
% above does.
boundAttrs = attributes;
boundAttrs{1}.r = 1; boundAttrs{1}.exch = true;
boundAttrs{2}.r = 1; boundAttrs{2}.exch = true;
bound = localConvert(g, boundAttrs);
boundAttr = unpackPreMaet(bound);
fprintf(['bound and role-free: %d x %d — a chord per event, but each ' ...
         'attribute\nunordered and so unpaired with the other\n'], ...
        size(boundAttr{1}, 1), size(boundAttr{1}, 2));

%% Local functions

function pm = localConvert(g, attributes, varargin)
    %localConvert The one conversion these encodings vary around.
    pm = preMaetFromAttrTable(g, 'attributes', attributes, ...
                          'time', 'beats', varargin{:});
end

function events = localChordEvents(pm, beat, perChord)
    %localChordEvents The events of the chord sounding at BEAT.
    [pAttr, ~, specs] = unpackPreMaet(pm);
    allNames = cellfun(@(s) s.name, specs, 'UniformOutput', false);
    onsets = pAttr{find(strcmp(allNames, 'onset'), 1)}(1, :);
    first = find(abs(onsets - beat) < 1e-9, 1);
    events = first:(first + perChord - 1);
end
