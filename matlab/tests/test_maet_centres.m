%% test_maet_centres.m — maetCentres, the public tuple-centre accessor
%
%  buildMaet builds the per-tuple fields lazily, so a density that
%  reached its consumer by the Mobius route carries no Centres array.
%  The accessor is the supported way to ask for one: it materialises the
%  fields when they are absent and passes them through when they are
%  not. What is pinned here is the shape and content of what it returns,
%  that it works on a lazy density, and that it rejects anything that is
%  not a density.
%
%  Mirror of Python tests/test_maet_centres.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

PERIOD = 1200;
TRIAD  = [0 400 700];

densAbs = buildMaet(TRIAD, [], 10, 2, false, false, PERIOD, ...
                    'verbose', false);
densRel = buildMaet(TRIAD, [], 10, 2, true, false, PERIOD, ...
                    'verbose', false);

% --- shape and content ----------------------------------------------------

C = maetCentres(densAbs);
results{end+1,1} = 'maetCentres: returns one matrix per attribute';
results{end,2}   = iscell(C) && isscalar(C);

results{end+1,1} = 'maetCentres: r = 2 absolute on three pitches is 2-by-6';
results{end,2}   = isequal(size(C{1}), [2 6]);

% The six ordered pairs of distinct elements, in the density's own
% coordinates.
expected = [  0 400;   0 700; ...
            400   0; 400 700; ...
            700   0; 700 400];
results{end+1,1} = 'maetCentres: the columns are the ordered index pairs';
results{end,2}   = isequal(sortrows(C{1}.'), sortrows(expected));

% A relative attribute is one coordinate shorter than its tuple size,
% the translation having been quotiented out.
results{end+1,1} = 'maetCentres: a relative attribute loses one coordinate';
Crel = maetCentres(densRel);
results{end,2}   = size(Crel{1}, 1) == 1;

% Unordered, the r-tuples are every arrangement of the event's elements;
% ordered, only the index-increasing ones.
pAttr = {TRIAD(:)};
specsUnord = flatSpecs(pAttr, 'r', 2, 'rel', false, 'exch', true);
specsOrd   = flatSpecs(pAttr, 'r', 2, 'rel', false, 'exch', false);
kw = {'sigma', 10, 'isPer', false, 'period', PERIOD, 'verbose', false};
Cunord = maetCentres(buildMaet(pAttr, [], 'specs', specsUnord, kw{:}));
Cord   = maetCentres(buildMaet(pAttr, [], 'specs', specsOrd, kw{:}));
results{end+1,1} = ['maetCentres: ordered tuples are the ' ...
                    'index-increasing subsequences'];
results{end,2}   = size(Cunord{1}, 2) == 6 && size(Cord{1}, 2) == 3;

% Each centre carries a kernel, so the density there is at least the
% weight of that one kernel.
vals = evalMaet(densAbs, C{1}, 'verbose', false);
results{end+1,1} = 'maetCentres: the density at its own centres is at least 1';
results{end,2}   = all(vals(:) >= 1 - 1e-9);

% --- laziness -------------------------------------------------------------

densLazy = buildMaet(TRIAD, [], 10, 2, false, false, PERIOD, ...
                     'verbose', false);
results{end+1,1} = 'maetCentres: buildMaet is lazy by default';
results{end,2}   = ~isfield(densLazy, 'Centres') || isempty(densLazy.Centres);

Clazy = maetCentres(densLazy);
results{end+1,1} = 'maetCentres: materialises a lazy density';
results{end,2}   = isequal(size(Clazy{1}), [2 6]);

densEager = buildMaet(TRIAD, [], 10, 2, false, false, PERIOD, ...
                      'lazy', false, 'verbose', false);
Ceager = maetCentres(densEager);
results{end+1,1} = 'maetCentres: passes an eager density through unchanged';
results{end,2}   = isequal(Ceager{1}, Clazy{1});

% --- validation -----------------------------------------------------------

results{end+1,1} = 'maetCentres: a numeric input errors';
results{end,2}   = throwsErrorWithId(@() maetCentres(TRIAD), ...
                                     'maetCentres:badInput');

results{end+1,1} = 'maetCentres: a struct without the tag errors';
results{end,2}   = throwsErrorWithId(@() maetCentres(struct('p', TRIAD)), ...
                                     'maetCentres:badInput');


if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf(['\n=== test_maet_centres: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_maet_centres:failed', '%d test(s) failed.', nFail);
    end
end
