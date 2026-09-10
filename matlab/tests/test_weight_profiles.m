%% test_weight_profiles.m — weight profiles beyond the window family
%
%  weightEvents accepts three kinds of profile: a numeric gamma (the
%  fixed-variance convolution family, tested in test_maet.m), one of
%  seven named profiles, and any function handle of the centred
%  difference. The named profiles absorb the serial-position curves
%  that seqWeights supplied before v3, so the values pinned here are
%  the ones that function produced: a rate of 1 over event numbers
%  1, ..., N decays by a factor of e per event.
%
%  Mirror of Python tests/test_weight_profiles.py; the two pin the same
%  numbers.
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

wpIdx   = [1 2 3 4];
wpPitch = [60 62 64 65];
wpTol   = 1e-12;

wpPm      = preMaet({wpPitch, wpIdx});
wpRecency = exp(-[3 2 1 0]);

%% --- The anchored serial-position profiles ---------------------------

results{end+1, 1} = 'weight profiles: exponentialFromEnd is the old recency curve';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, NaN, 'exponentialFromEnd') - wpRecency)) < wpTol;

results{end+1, 1} = 'weight profiles: exponentialFromStart mirrors it';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, NaN, 'exponentialFromStart') ...
            - fliplr(wpRecency))) < wpTol;

results{end+1, 1} = 'weight profiles: decayRate and sd are two spellings of one scale';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, NaN, 'exponentialFromEnd', 'decayRate', 0.5) ...
                            - localProfileWeights(wpPm, NaN, ...
                                  'exponentialFromEnd', 'sd', 2))) < wpTol;

wpU = localProfileWeights(wpPm, NaN, 'uShape');
results{end+1, 1} = 'weight profiles: uShape is symmetric, and alpha selects its components';
results{end, 2}   = max(abs(wpU - fliplr(wpU))) < wpTol ...
    && max(abs(localProfileWeights(wpPm, NaN, 'uShape', 'alpha', 1) ...
               - localProfileWeights(wpPm, NaN, 'exponentialFromStart'))) < wpTol ...
    && max(abs(localProfileWeights(wpPm, NaN, 'uShape', 'alpha', 0) ...
               - localProfileWeights(wpPm, NaN, 'exponentialFromEnd'))) < wpTol;

wpExpectedAsym = 0.4 * exp(-0.2 * (wpIdx - wpIdx(1))) ...
    + 0.6 * exp(-0.8 * (wpIdx(end) - wpIdx));
results{end+1, 1} = 'weight profiles: uAsym gives its two components their own rates';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, NaN, 'uAsym', 'decayRateStart', 0.2, ...
                                'decayRateEnd', 0.8, 'alpha', 0.4) ...
                            - wpExpectedAsym)) < wpTol;

results{end+1, 1} = 'weight profiles: uAsym rates fall back to decayRate';
results{end, 2}   = max(abs(localProfileWeights(wpPm, NaN, 'uAsym', 'decayRate', 0.5) ...
                            - localProfileWeights(wpPm, NaN, 'uAsym', ...
                                  'decayRateStart', 0.5, ...
                                  'decayRateEnd', 0.5))) < wpTol;

%% --- The centre-anchored profiles ------------------------------------

results{end+1, 1} = 'weight profiles: exponentialBefore at the last event matches the anchored form';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, 4, 'exponentialBefore') - wpRecency)) < wpTol;

wpAfter = localProfileWeights(wpPm, 2, 'exponentialAfter');
results{end+1, 1} = 'weight profiles: exponentialAfter is zero below its centre';
results{end, 2}   = wpAfter(1) == 0 ...
    && max(abs(wpAfter(2:end) - exp(-[0 1 2]))) < wpTol;

results{end+1, 1} = 'weight profiles: the two-sided exponential holds its standard deviation';
results{end, 2}   = max(abs(localProfileWeights(wpPm, 2, 'exponential', 'sd', 1) ...
                            - exp(-abs(wpIdx - 2) * sqrt(2)))) < wpTol;

%% --- Function handles -------------------------------------------------

results{end+1, 1} = 'weight profiles: a function handle supplies any profile';
results{end, 2}   = max(abs(localProfileWeights(wpPm, 4, @(d) exp(0.5 * d) .* (d <= 0)) ...
                            - exp(0.5 * [-3 -2 -1 0]))) < wpTol;

results{end+1, 1} = 'weight profiles: a function handle is exempt from the kernel truncation';
results{end, 2}   = ...
    max(abs(localProfileWeights(wpPm, 1, @(d) 1 + abs(d)) - (1 + abs(wpIdx - 1)))) < wpTol;

%% --- Rejected combinations -------------------------------------------

results{end+1, 1} = 'weight profiles: a centre is refused by the anchored profiles';
results{end, 2}   = ...
    throwsErrorWithId(@() localProfileWeights(wpPm, 4, 'exponentialFromEnd'), ...
    'weightEvents:centreWithAnchoredProfile');

results{end+1, 1} = 'weight profiles: width is refused by the named profiles';
results{end, 2}   = ...
    throwsErrorWithId(@() localProfileWeights(wpPm, NaN, ...
        'exponentialFromEnd', 'width', 2), ...
    'weightEvents:widthWithNamedProfile');

results{end+1, 1} = 'weight profiles: sd and decayRate together are refused';
results{end, 2}   = throwsErrorWithId( ...
    @() localProfileWeights(wpPm, NaN, 'exponentialFromEnd', 'sd', 1, 'decayRate', 1), ...
    'weightEvents:sdRateXor');

results{end+1, 1} = 'weight profiles: asymmetric rates are refused by the one-rate profiles';
results{end, 2}   = throwsErrorWithId( ...
    @() localProfileWeights(wpPm, NaN, 'uShape', 'decayRateStart', 0.5), ...
    'weightEvents:asymRatesWithSymmetricProfile');

results{end+1, 1} = 'weight profiles: a rate is refused by the numeric family';
results{end, 2}   = ...
    throwsErrorWithId(@() localProfileWeights(wpPm, 2, 0, 'sd', 1, 'decayRate', 1), ...
    'weightEvents:rateWithNumericShape');

results{end+1, 1} = 'weight profiles: a scale is refused with a function handle';
results{end, 2}   = throwsErrorWithId( ...
    @() localProfileWeights(wpPm, 2, @(d) ones(size(d)), 'sd', 1), ...
    'weightEvents:scaleWithProfile');

results{end+1, 1} = 'weight profiles: an unknown profile name is refused';
results{end, 2}   = ...
    throwsErrorWithId(@() localProfileWeights(wpPm, NaN, 'exponentialFromMiddle'), ...
    'weightEvents:badShapeName');

results{end+1, 1} = 'weight profiles: a handle must return one non-negative factor per event';
results{end, 2}   = throwsErrorWithId(@() localProfileWeights(wpPm, 2, @(d) ones(1, 2)), ...
    'weightEvents:profileSize') ...
    && throwsErrorWithId(@() localProfileWeights(wpPm, 2, @(d) -ones(size(d))), ...
    'weightEvents:profileNotNonNegative');

%% --- Composition ------------------------------------------------------

wpPm = weightEvents(preMaet({wpPitch, wpIdx}), 2, 1, NaN, ...
                    'exponentialFromEnd', 'dropInputAttr', true);
[wpPOut, wpWOut, wpSpecsOut] = unpackPreMaet(wpPm);
results{end+1, 1} = 'weight profiles: the driving attribute is dropped and the pitches survive';
results{end, 2}   = numel(wpPOut) == 1 && numel(wpSpecsOut) == 1 ...
    && isequal(wpPOut{1}, wpPitch) ...
    && max(abs(wpWOut{1} - wpRecency)) < wpTol;

wpSalience = [1 0.6 0.6 1.3];
wpPmS = weightEvents(preMaet({wpPitch, wpIdx}, {wpSalience, []}), 2, 1, NaN, ...
                     'exponentialFromEnd', 'dropInputAttr', true);
[~, wpWS, ~] = unpackPreMaet(wpPmS);
results{end+1, 1} = 'weight profiles: an existing weight is multiplied into, not replaced';
results{end, 2}   = max(abs(wpWS{1} - wpSalience .* wpRecency)) < wpTol;

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
    fprintf('\n=== test_weight_profiles: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_weight_profiles:failed', '%d test(s) failed.', nFail);
    end
end

function w = localProfileWeights(pm0, centre, shape, varargin)
%LOCALPROFILEWEIGHTS  Weight profile applied to pitch via an event number.
%   Drives a weightEvents profile on the two-attribute pre-MAET PM0
%   (pitch, event number) from the event-number attribute, drops that
%   attribute, and returns the pitch weights.
    pm = weightEvents(pm0, 2, 1, centre, shape, ...
                      'dropInputAttr', true, varargin{:});
    [~, wOut, ~] = unpackPreMaet(pm);
    w = wOut{1};
end
