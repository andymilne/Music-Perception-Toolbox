%% test_batched_shape_rule.m — the shape rule that selects batched-raw mode
%
%  MATLAB dispatches on a matrix with both dimensions greater than one,
%  so an M-by-1 column is a vector and is broadcast as one multiset
%  shared by every row. Python dispatches on ndim, where an (M, 1) array
%  is M rows of one element each. The two rules are each idiomatic in
%  their own language and are documented as a deliberate divergence in
%  the entry-point docstrings; these tests pin the MATLAB side of it,
%  and pin the NaN-padded spelling that reads the same way in both
%  languages.
%
%  Mirror of Python tests/test_batched_shape_rule.py.
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
SIGMA  = 10;
SCALE  = [0 200 400 500 700 900 1100];
PROBES = [0 100 200 300];

nRows   = numel(PROBES);
context = repmat(SCALE, nRows, 1);

% --- a column is a vector, not a batch ------------------------------------

sCol = simMaet(context, [], PROBES(:), [], SIGMA, 1, false, true, ...
               PERIOD, 'verbose', false);
results{end+1,1} = ['simMaet: a column operand is broadcast as one ' ...
                    'shared multiset'];
results{end,2}   = numel(sCol) == nRows ...
                   && (max(sCol(:)) - min(sCol(:))) < 1e-12;

sRow = simMaet(context, [], PROBES, [], SIGMA, 1, false, true, ...
               PERIOD, 'verbose', false);
results{end+1,1} = 'simMaet: row and column operands agree, both broadcasting';
results{end,2}   = max(abs(sCol(:) - sRow(:))) < 1e-12;

% --- the NaN-padded spelling ----------------------------------------------

probes = [PROBES(:), nan(nRows, 1)];
sPad = simMaet(context, [], probes, [], SIGMA, 1, false, true, ...
               PERIOD, 'verbose', false);
results{end+1,1} = 'simMaet: NaN padding gives one result per probe row';
results{end,2}   = numel(sPad) == nRows ...
                   && (max(sPad(:)) - min(sPad(:))) > 1e-3;

% The padding is stripped per row, so each row is the one-tone probe it
% was written to be: the profile is the same as comparing each probe
% separately.
sOneByOne = zeros(nRows, 1);
for i = 1:nRows
    sOneByOne(i) = simMaet(SCALE, [], PROBES(i), [], SIGMA, 1, ...
                           false, true, PERIOD, 'verbose', false);
end
results{end+1,1} = 'simMaet: the padded rows match probe-by-probe calls';
results{end,2}   = max(abs(sPad(:) - sOneByOne)) < 1e-10;

% The same holds with a spectrum applied inside the call, which is how
% the probe-tone demo uses it.
SPECTRUM = {'harmonic', 8, 'powerlaw', 1};
sPadSpec = simMaet(context, [], probes, [], SIGMA, 1, false, true, ...
                   PERIOD, 'spectrum', SPECTRUM, 'verbose', false);
% The per-probe reference is itself a batched call, of two identical
% rows. 'spectrum' is a batched-raw name-value in MATLAB -- on a scalar
% operand the caller is told to apply addSpectra beforehand -- so a
% one-probe-at-a-time loop cannot express the comparison here, and the
% smallest batch that can is a pair.
sSpecOneByOne = zeros(nRows, 1);
for i = 1:nRows
    sPair = simMaet(repmat(SCALE, 2, 1), [], ...
                    [PROBES(i) NaN; PROBES(i) NaN], [], SIGMA, 1, ...
                    false, true, PERIOD, ...
                    'spectrum', SPECTRUM, 'verbose', false);
    sSpecOneByOne(i) = sPair(1);
end
results{end+1,1} = 'simMaet: NaN padding survives the spectrum name-value';
results{end,2}   = max(abs(sPadSpec(:) - sSpecOneByOne)) < 1e-10;

% --- the same rule in evalMaet and entropyMaet ----------------------------

X = [0 100 200];
valsCol = evalMaet(PROBES(:), [], SIGMA, 1, false, true, PERIOD, X, ...
                   'verbose', false);
results{end+1,1} = ['evalMaet: a column takes the single multiset raw ' ...
                    'path'];
results{end,2}   = isequal(size(valsCol), [1 numel(X)]);

valsPad = evalMaet([PROBES(:), nan(nRows, 1)], [], SIGMA, 1, false, ...
                   true, PERIOD, X, 'verbose', false);
results{end+1,1} = 'evalMaet: NaN padding gives one row per probe';
results{end,2}   = isequal(size(valsPad), [nRows numel(X)]);

HCol = entropyMaet(PROBES(:), [], SIGMA, 1, false, true, PERIOD, ...
                   'nPointsPerDim', 120, 'verbose', false);
results{end+1,1} = 'entropyMaet: a column takes the single multiset raw path';
results{end,2}   = isscalar(HCol);

HPad = entropyMaet([PROBES(:), nan(nRows, 1)], [], SIGMA, 1, false, ...
                   true, PERIOD, 'nPointsPerDim', 120, 'verbose', false);
results{end+1,1} = 'entropyMaet: NaN padding gives one entropy per probe';
results{end,2}   = numel(HPad) == nRows;


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
    fprintf(['\n=== test_batched_shape_rule: %d passed, %d failed ' ...
             '(of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_batched_shape_rule:failed', '%d test(s) failed.', nFail);
    end
end
