%% test_abs_per_single_image_warning.m
%  Tests for the absolute-periodic single-image warning. Mirrors the
%  Python tests/test_abs_per_single_image_warning.py.
%
%  The absolute-periodic kernel wraps each difference to its nearest
%  image. Below sigma/period = 0.05 that agrees with the full-image
%  measure to within the toolbox's own accuracy floor; above it the two
%  diverge, and above roughly 0.15 the single-image kernel also stops
%  being positive definite. The warning marks the crossing.
%
%  Raised at density construction, because the choice is a property of
%  the density rather than of any one operation.
%
%  No local functions: this file is executed as a script from
%  test_mpt.m, so the warning probe is written inline.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

absPerP  = 1200;
absPerID = 'buildExpTens:absPerSingleImage';
absPerP_ = [0; 100; 300; 700];
absPerW_ = ones(4, 1);

% Each row: label, sigma/period, isRel, isPer, period, expectWarning
absPerCases = { ...
    'sigma/P=0.06 -> warns',        0.06,   false, true,  absPerP, true ; ...
    'sigma/P=0.10 -> warns',        0.10,   false, true,  absPerP, true ; ...
    'sigma/P=0.20 -> warns',        0.20,   false, true,  absPerP, true ; ...
    'sigma/P=0.30 -> warns',        0.30,   false, true,  absPerP, true ; ...
    'sigma/P=0.50 -> warns',        0.50,   false, true,  absPerP, true ; ...
    'sigma/P=0.001 -> silent',      0.001,  false, true,  absPerP, false; ...
    'sigma/P=0.0125 -> silent',     0.0125, false, true,  absPerP, false; ...
    'sigma/P=0.02 -> silent',       0.02,   false, true,  absPerP, false; ...
    'sigma/P=0.03 -> silent',       0.03,   false, true,  absPerP, false; ...
    'sigma/P=0.04 -> silent',       0.04,   false, true,  absPerP, false; ...
    'sigma/P=0.05 exactly -> silent', 0.05, false, true,  absPerP, false; ...
    'rel-per -> silent',            0.30,   true,  true,  absPerP, false; ...
    'abs-non-per -> silent',        0.30,   false, false, 0,       false; ...
    'rel-non-per -> silent',        0.30,   true,  false, 0,       false; ...
};

absPerPrev = warning('off', absPerID);
for ii = 1:size(absPerCases, 1)
    sop    = absPerCases{ii, 2};
    isRel_ = absPerCases{ii, 3};
    isPer_ = absPerCases{ii, 4};
    period_= absPerCases{ii, 5};
    expect = absPerCases{ii, 6};
    lastwarn('', '');
    fired = false;
    try
        buildExpTens(absPerP_, absPerW_, sop * absPerP, 2, ...
                     isRel_, isPer_, period_);
        [~, lastId] = lastwarn();
        fired = strcmp(lastId, absPerID);
    catch
        fired = false;
    end
    results(end+1, :) = { ...
        ['absPer warning: ' absPerCases{ii, 1}], ...
        isequal(fired, expect)}; %#ok<*SAGROW>
end

% Construction must still succeed at every r, warning or not.
absPerOK = true;
try
    for r = 1:3
        d = buildExpTens(absPerP_, absPerW_, 0.20 * absPerP, r, ...
                         false, true, absPerP);
        absPerOK = absPerOK && isstruct(d);
    end
catch
    absPerOK = false;
end
results(end+1, :) = {'absPer warning: construction succeeds at r=1..3', ...
                     absPerOK};
warning(absPerPrev);


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_abs_per_single_image_warning: %d passed, %d failed\n', ...
            nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end
