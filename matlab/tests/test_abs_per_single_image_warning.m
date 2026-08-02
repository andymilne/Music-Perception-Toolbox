%% test_abs_per_single_image_warning.m
%  Tests for the absolute-periodic single-image warning. Mirrors the
%  Python tests/test_abs_per_single_image_warning.py.
%
%  The absolute-periodic kernel wraps each difference to its nearest
%  image, approximating the full-image measure that sums over every
%  image. The two agree while sigma/period is small and diverge as it
%  grows; past the threshold in internal.absPerSigmaOverPThreshold the
%  single-image kernel also stops being positive definite, so the
%  similarity it induces is not bounded by 1. The boundary cases below
%  read that threshold rather than restating its value.
%
%  In v3+ the warning only fires when the user has explicitly opted an
%  abs-per attribute into ``wrap = 'single-image'``. The default
%  ``wrap = 'full-image'`` measure is positive definite by construction
%  and stays silent.
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

% Each row: label, sigma/period, isRel, isPer, period, wrap, expectWarning
%
% Cases where wrap='single-image' at high sigma/P: warn (single-image
% opt-in is genuinely a choice with consequences).
% Cases where wrap defaults to full-image: stay silent (full-image is
% PD and needs no warning).
absPerThr = internal.absPerSigmaOverPThreshold();
absPerCases = { ...
    'sigma/P=0.06 opted single -> warns',  0.06, false, true,  absPerP, 'single-image', true ; ...
    'sigma/P=0.10 opted single -> warns',  0.10, false, true,  absPerP, 'single-image', true ; ...
    'sigma/P=0.20 opted single -> warns',  0.20, false, true,  absPerP, 'single-image', true ; ...
    'sigma/P=0.30 opted single -> warns',  0.30, false, true,  absPerP, 'single-image', true ; ...
    'sigma/P=0.50 opted single -> warns',  0.50, false, true,  absPerP, 'single-image', true ; ...
    'sigma/P=0.30 default full -> silent', 0.30, false, true,  absPerP, [],             false; ...
    'sigma/P=0.50 default full -> silent', 0.50, false, true,  absPerP, [],             false; ...
    'sigma/P=0.001 -> silent',             0.001, false, true, absPerP, [],             false; ...
    'sigma/P=0.0125 -> silent',            0.0125, false, true,absPerP, [],             false; ...
    'sigma/P=0.02 -> silent',              0.02, false, true,  absPerP, [],             false; ...
    'sigma/P=0.03 -> silent',              0.03, false, true,  absPerP, [],             false; ...
    'rel-per -> silent',                   0.30, true,  true,  absPerP, [],             false; ...
    'abs-non-per -> silent',               0.30, false, false, 0,       [],             false; ...
    'rel-non-per -> silent',               0.30, true,  false, 0,       [],             false; ...
    % Boundary, read from the threshold itself rather than written out:
    % the comparison is strict, so the threshold value does not warn and
    % anything above it does. Spelling the numbers here let this table
    % and the warning drift apart when the threshold moved.
    'well below the threshold, opted single -> silent', absPerThr / 2, ...
                                           false, true, absPerP, ...
                                           'single-image',          false; ...
    'at the threshold, opted single -> silent', absPerThr, false, true, ...
                                           absPerP, 'single-image', false; ...
    'above the threshold, opted single -> warns', absPerThr * 1.25, ...
                                           false, true, absPerP, ...
                                           'single-image',          true ; ...
};

absPerPrev = warning('off', absPerID);
for ii = 1:size(absPerCases, 1)
    sop    = absPerCases{ii, 2};
    isRel_ = absPerCases{ii, 3};
    isPer_ = absPerCases{ii, 4};
    period_= absPerCases{ii, 5};
    wrap_  = absPerCases{ii, 6};
    expect = absPerCases{ii, 7};
    lastwarn('', '');
    fired = false;
    try
        if isempty(wrap_)
            buildExpTens(absPerP_, absPerW_, sop * absPerP, 2, ...
                         isRel_, isPer_, period_);
        else
            buildExpTens(absPerP_, absPerW_, sop * absPerP, 2, ...
                         isRel_, isPer_, period_, 'wrap', wrap_);
        end
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
