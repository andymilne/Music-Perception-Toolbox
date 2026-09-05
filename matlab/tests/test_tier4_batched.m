%% test_tier4_batched.m — Tier-4 batched dispatch (v3+): balanceCircular, evennessCircular
%
%  Tests for Tier-4 batched dispatch (v3+): balanceCircular, evennessCircular.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end


%
% These Monte-Carlo functions accept an nRows-by-K matrix and return
% per-row column vectors. The new ``rngScope`` NV pair controls how
% each row's RNG seed is derived from the base ``rngSeed``:
%   'canonical' (default): derived from canonical-form key, so
%      transposition-equivalent rows share an MC realisation; dedup
%      works for sigma > 0.
%   'row': derived from row index, so identical rows get distinct
%      reproducible realisations; dedup is disabled.

% --- balanceCircular: sigma = 0 (deterministic) ---
P_b0 = [0, 4, 7; 0, 3, 7; 0, 4, 7];
bVec0 = balanceCircular(P_b0, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: returns nRows-by-1 vector';
results{end,2}   = isnumeric(bVec0) && isequal(size(bVec0), [3, 1]);

results{end+1,1} = 'balanceCircular batched sigma=0: matches scalar dispatch';
results{end,2}   = abs(bVec0(2) - balanceCircular(P_b0(2, :)', [], 12, 0)) < 1e-12;

results{end+1,1} = 'balanceCircular batched sigma=0: dedup (rows 1, 3 identical)';
results{end,2}   = bVec0(1) == bVec0(3);

[bVec0_, bStd0_] = balanceCircular(P_b0, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: bStd is zero with two outputs';
results{end,2}   = isequal(size(bStd0_), [3, 1]) && all(bStd0_ == 0);

% NaN-padded
P_b0_nan = [0, 4, 7, NaN; 0, 1, 5, 6; NaN, NaN, NaN, NaN];
bNan = balanceCircular(P_b0_nan, [], 12, 0);
results{end+1,1} = 'balanceCircular batched sigma=0: NaN-padded rows give NaN';
results{end,2}   = ~isnan(bNan(1)) && ~isnan(bNan(2)) && isnan(bNan(3));

% --- balanceCircular: sigma > 0 (Monte Carlo) ---

% Canonical scope: identical inputs give identical MC results
P_b1 = [0, 4, 7; 4, 0, 7; 0, 4, 7];
bCanon = balanceCircular(P_b1, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'balanceCircular batched MC canonical: identical inputs identical results';
results{end,2}   = bCanon(1) == bCanon(2) && bCanon(1) == bCanon(3);

% Different canonicals -> different results
P_b2 = [0, 4, 7; 0, 3, 7];
bDiff = balanceCircular(P_b2, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'balanceCircular batched MC canonical: different scales give different results';
results{end,2}   = bDiff(1) ~= bDiff(2);

% Reproducibility across calls
b_call_a = balanceCircular(P_b1, [], 12, 0.3, 'rngSeed', 42);
b_call_b = balanceCircular(P_b1, [], 12, 0.3, 'rngSeed', 42);
results{end+1,1} = 'balanceCircular batched MC: reproducible with same rngSeed';
results{end,2}   = isequal(b_call_a, b_call_b);

% Row scope: identical inputs give DIFFERENT realisations
bRow = balanceCircular(P_b1, [], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'row');
results{end+1,1} = 'balanceCircular batched MC row scope: identical inputs differ';
results{end,2}   = bRow(1) ~= bRow(2) && bRow(2) ~= bRow(3) && bRow(1) ~= bRow(3);

% Empty rngSeed in batched: within-call dedup still works
bDedup = balanceCircular(P_b1, [], 12, 0.3);
results{end+1,1} = 'balanceCircular batched MC: within-call dedup with empty rngSeed';
results{end,2}   = bDedup(1) == bDedup(2) && bDedup(1) == bDedup(3);

% --- evennessCircular: brief MC sanity ---
P_e = [0, 4, 7; 4, 0, 7];
eDedup = evennessCircular(P_e, 12, 0.3, 'rngSeed', 42, 'rngScope', 'canonical');
results{end+1,1} = 'evennessCircular batched MC canonical: dedup works';
results{end,2}   = eDedup(1) == eDedup(2);

eRow = evennessCircular([0, 4, 7; 0, 4, 7], 12, 0.3, ...
    'rngSeed', 42, 'rngScope', 'row');
results{end+1,1} = 'evennessCircular batched MC row scope: identical inputs differ';
results{end,2}   = eRow(1) ~= eRow(2);


%% ---- Standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_tier4_batched: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_tier4_batched:failed', '%d test(s) failed.', nFail);
    end
end
