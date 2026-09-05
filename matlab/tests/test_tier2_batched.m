%% test_tier2_batched.m — Tier-2 batched dispatch (v3+): coherence, sameness, nTupleEntropy
%
%  Tests for Tier-2 batched dispatch (v3+): coherence, sameness, nTupleEntropy.
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
% These set-based functions accept an nRows-by-K matrix and return
% per-row results: nRows-by-1 vectors for scalar outputs and 1-by-nRows
% cell arrays for variable-shape outputs (nTupleEntropy's ``tuples``).
% Per-row dedup uses a sorted-modular canonical key (permutation +
% period symmetries; not transposition).

% --- coherence ---
P_coh = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[cVec, ncVec] = coherence(P_coh, 12);
results{end+1,1} = 'coherence batched: returns nRows-by-1 vectors';
results{end,2}   = isnumeric(cVec) && isequal(size(cVec), [3, 1]) ...
                && isequal(size(ncVec), [3, 1]);

[c_scalar, nc_scalar] = coherence(P_coh(2, :)', 12);
results{end+1,1} = 'coherence batched: matches scalar dispatch';
results{end,2}   = abs(cVec(2) - c_scalar) < 1e-12 ...
                && abs(ncVec(2) - nc_scalar) < 1e-12;

results{end+1,1} = 'coherence batched: dedup row 1 = row 3';
results{end,2}   = cVec(1) == cVec(3) && ncVec(1) == ncVec(3);

% Permutation dedup
P_coh_perm = [0, 4, 7; 4, 0, 7];
[cPerm, ~] = coherence(P_coh_perm, 12);
results{end+1,1} = 'coherence batched: permutation dedup';
results{end,2}   = cPerm(1) == cPerm(2);

% NaN-padded
P_coh_pad = [0, 4, 7, NaN; 0, 1, 5, 6; NaN, NaN, NaN, NaN];
[cPad, ~] = coherence(P_coh_pad, 12);
results{end+1,1} = 'coherence batched: NaN-padded variable cardinality';
results{end,2}   = ~isnan(cPad(1)) && ~isnan(cPad(2)) && isnan(cPad(3));

% --- sameness ---
P_sm = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[sqVec, ndVec] = sameness(P_sm, 12);
results{end+1,1} = 'sameness batched: returns nRows-by-1 vectors';
results{end,2}   = isequal(size(sqVec), [3, 1]);

[sq_s, nd_s] = sameness(P_sm(2, :)', 12);
results{end+1,1} = 'sameness batched: matches scalar dispatch';
results{end,2}   = abs(sqVec(2) - sq_s) < 1e-12 ...
                && abs(ndVec(2) - nd_s) < 1e-12;

results{end+1,1} = 'sameness batched: dedup row 1 = row 3';
results{end,2}   = sqVec(1) == sqVec(3);

% --- nTupleEntropy ---
P_nte = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[HVec, tuplesCell] = nTupleEntropy(P_nte, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: returns vector + cell';
results{end,2}   = isnumeric(HVec) && isequal(size(HVec), [3, 1]) ...
                && iscell(tuplesCell) && isequal(size(tuplesCell), [1, 3]);

[H_s, t_s] = nTupleEntropy(P_nte(2, :)', 12, 1);
results{end+1,1} = 'nTupleEntropy batched: matches scalar dispatch';
results{end,2}   = abs(HVec(2) - H_s) < 1e-12 ...
                && isequal(tuplesCell{2}, t_s);

results{end+1,1} = 'nTupleEntropy batched: dedup row 1 = row 3';
results{end,2}   = HVec(1) == HVec(3);

% NaN-padded all-NaN row gives NaN H and empty tuples
P_nte_pad = [0, 4, 7, NaN; NaN, NaN, NaN, NaN];
[Hpad, tpad] = nTupleEntropy(P_nte_pad, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: all-NaN row returns NaN';
results{end,2}   = ~isnan(Hpad(1)) && isnan(Hpad(2)) && isempty(tpad{2});

% --- Transposition dedup (necklace canonical) for coherence and sameness ---
% All 12 transpositions of the major triad share one necklace key, so
% values are identical across rows.
shifts = (0:11)';
P_majorTrans = mod([0, 4, 7] + shifts, 12);  % 12-by-3
[cTrans, ncTrans] = coherence(P_majorTrans, 12);
results{end+1,1} = 'coherence batched: all 12 major-triad transpositions agree';
results{end,2}   = all(cTrans == cTrans(1)) && all(ncTrans == ncTrans(1));

% Coherence with failures: {0, 1, 5, 6} and 12 transpositions
P_aug = mod([0, 1, 5, 6] + shifts, 12);
[cAug, ncAug] = coherence(P_aug, 12);
results{end+1,1} = 'coherence batched: transposition dedup with non-trivial nc';
results{end,2}   = all(cAug == cAug(1)) && ncAug(1) == 5.0;

% Sameness: 12 major-triad transpositions agree
[sqTrans, ndTrans] = sameness(P_majorTrans, 12);
results{end+1,1} = 'sameness batched: all 12 major-triad transpositions agree';
results{end,2}   = all(sqTrans == sqTrans(1)) && all(ndTrans == ndTrans(1));

% nTupleEntropy: H is transposition-invariant; tuples is NOT (in general).
% Pin this down to make the dedup contract explicit.
P_nteTrans = [0, 4, 7; 2, 7, 11];   % {0,4,7} and its shift-by-7 transposition
[HnteTrans, tnteTrans] = nTupleEntropy(P_nteTrans, 12, 1);
results{end+1,1} = 'nTupleEntropy batched: H is transposition-invariant';
results{end,2}   = abs(HnteTrans(1) - HnteTrans(2)) < 1e-12;
results{end+1,1} = 'nTupleEntropy batched: tuples differ across transpositions';
results{end,2}   = ~isequal(tnteTrans{1}, tnteTrans{2});


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
    fprintf('\n=== test_tier2_batched: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_tier2_batched:failed', '%d test(s) failed.', nFail);
    end
end
