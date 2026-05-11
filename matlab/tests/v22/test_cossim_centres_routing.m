function tests = test_cossim_centres_routing
%TEST_COSSIM_CENTRES_ROUTING  Stage 2b: cosSimExpTens centres-IP routing.
%
%   Verifies that cosSimExpTens routes the SA centres-IP through
%   internal.gaussianKernelSum for the abs (±periodic) and rel
%   non-periodic modes, and that the truncationSigmas / kernelPrecision
%   kwargs reach the helper from the public entry point.
%
%   Coverage:
%     - cosSimExpTens(dens_x, dens_y, 'truncationSigmas', k) returns the
%       same value as the exact path for abs ±per and rel non-per, to
%       within the helper's truncation tolerance.
%     - The rel+periodic case (pairwise-wrap quadratic form) stays on
%       the existing pairwise code path; the kwarg is accepted without
%       error but doesn't change the result.
%     - Global mptDefaults('truncationSigmas', k) propagates.

    tests = functiontests(localfunctions);
end


% =========================================================================
%  Helpers
% =========================================================================

function [dens_x, dens_y] = buildPair(isRel, isPer)
    sigma = 12;
    period = 1200;
    r = 3;
    p_x = [0, 400, 700];
    p_y = [0, 300, 700];
    if isPer
        Jval = period;
    else
        Jval = 0;
    end
    dens_x = buildExpTens(p_x(:), ones(3, 1), sigma, r, ...
        isRel, isPer, Jval, 'verbose', false);
    dens_y = buildExpTens(p_y(:), ones(3, 1), sigma, r, ...
        isRel, isPer, Jval, 'verbose', false);
end


% =========================================================================
%  Exact vs truncated across non-rel-per modes
% =========================================================================

function test_abs_nonper_exact_vs_truncated(testCase)
    mptDefaults('reset');
    [dens_x, dens_y] = buildPair(false, false);
    s_exact = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'verbose', false);
    s_trunc = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'truncationSigmas', 6, 'verbose', false);
    verifyLessThan(testCase, abs(s_exact - s_trunc), 1e-7, ...
        sprintf('abs+nonper: exact=%g trunc=%g diff=%.3e', ...
            s_exact, s_trunc, abs(s_exact - s_trunc)));
end

function test_abs_per_exact_vs_truncated(testCase)
    mptDefaults('reset');
    [dens_x, dens_y] = buildPair(false, true);
    s_exact = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'verbose', false);
    s_trunc = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'truncationSigmas', 6, 'verbose', false);
    verifyLessThan(testCase, abs(s_exact - s_trunc), 1e-7, ...
        sprintf('abs+per: exact=%g trunc=%g diff=%.3e', ...
            s_exact, s_trunc, abs(s_exact - s_trunc)));
end

function test_rel_nonper_exact_vs_truncated(testCase)
    mptDefaults('reset');
    [dens_x, dens_y] = buildPair(true, false);
    s_exact = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'verbose', false);
    s_trunc = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'truncationSigmas', 6, 'verbose', false);
    verifyLessThan(testCase, abs(s_exact - s_trunc), 1e-7, ...
        sprintf('rel+nonper: exact=%g trunc=%g diff=%.3e', ...
            s_exact, s_trunc, abs(s_exact - s_trunc)));
end


% =========================================================================
%  Rel+per: helper not used, kwarg accepted but ignored
% =========================================================================

function test_rel_per_unaffected_by_truncation_kwarg(testCase)
    mptDefaults('reset');
    [dens_x, dens_y] = buildPair(true, true);
    s_default = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'verbose', false);
    s_with = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'truncationSigmas', 6, 'verbose', false);
    verifyEqual(testCase, s_default, s_with, ...
        'rel+per: kwarg should not affect output (existing pairwise-wrap path)');
end


% =========================================================================
%  Global default propagates
% =========================================================================

function test_global_default_propagates(testCase)
    mptDefaults('reset');
    [dens_x, dens_y] = buildPair(true, false);
    s_explicit = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'truncationSigmas', 6, 'verbose', false);
    mptDefaults('truncationSigmas', 6);
    cleanupObj = onCleanup(@() mptDefaults('reset'));
    s_global = cosSimExpTens(dens_x, dens_y, ...
        'method', 'pairwise', 'verbose', false);
    verifyEqual(testCase, s_explicit, s_global, 'AbsTol', 0, ...
        'Global default should match explicit kwarg bit-exactly');
end
