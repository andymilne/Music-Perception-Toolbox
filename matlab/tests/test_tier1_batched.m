%% test_tier1_batched.m — Tier-1 batched dispatch (v2.1+): dftCircular, meanOffset, edges, projCentroid, circApm
%
%  Tests for Tier-1 batched dispatch (v2.1+): dftCircular, meanOffset, edges, projCentroid, circApm.
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
% Each accepts an nRows-by-K matrix in addition to the original 1-D
% form, and returns 1-by-nRows cell arrays. Per-row dedup is over
% permutation + period symmetries (not transposition).

% --- dftCircular ---
P_dft = [0, 200, 400, 500, 700, 900, 1100;     % major
         0, 100, 300, 500, 700, 900, 1000;     % something else
         0, 200, 400, 500, 700, 900, 1100];    % = row 1

[FCell, magCell] = dftCircular(P_dft, [], 1200);
results{end+1,1} = 'dftCircular batched: returns 1-by-nRows cells';
results{end,2}   = iscell(FCell) && iscell(magCell) ...
                && isequal(size(FCell), [1, 3]) ...
                && isequal(size(magCell), [1, 3]);

% Per-row matches scalar
[F_scalar, mag_scalar] = dftCircular(P_dft(2, :), [], 1200);
results{end+1,1} = 'dftCircular batched: matches scalar dispatch row-by-row';
results{end,2}   = max(abs(FCell{2} - F_scalar)) < 1e-12 ...
                && max(abs(magCell{2} - mag_scalar)) < 1e-12;

% Permutation dedup
P_perm = [0, 200, 400; 400, 0, 200];
[Fperm, magPerm] = dftCircular(P_perm, [], 1200);
results{end+1,1} = 'dftCircular batched: permutation dedup';
results{end,2}   = isequal(Fperm{1}, Fperm{2});

% NaN-padded variable cardinality
P_nan = [0, 200, 400, NaN; 0, 100, 200, 300; NaN, NaN, NaN, NaN];
[Fnan, magNan] = dftCircular(P_nan, [], 1200);
results{end+1,1} = 'dftCircular batched: NaN-padded cardinality OK';
results{end,2}   = numel(Fnan{1}) == 3 && numel(Fnan{2}) == 4 && isempty(Fnan{3});

% --- meanOffset ---
P_mo = [0, 4, 7; 0, 3, 7; 0, 4, 7];
hMOCell = meanOffset(P_mo, [], 12);
results{end+1,1} = 'meanOffset batched: returns 1-by-nRows cell';
results{end,2}   = iscell(hMOCell) && isequal(size(hMOCell), [1, 3]);

hMOscalar = meanOffset(P_mo(2, :), [], 12);
results{end+1,1} = 'meanOffset batched: matches scalar dispatch';
results{end,2}   = max(abs(hMOCell{2} - hMOscalar)) < 1e-12;

results{end+1,1} = 'meanOffset batched: dedup row 1 = row 3';
results{end,2}   = isequal(hMOCell{1}, hMOCell{3});

% --- edges ---
P_e = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[eCell, esCell] = edges(P_e, [], 12);
results{end+1,1} = 'edges batched: returns two 1-by-nRows cells';
results{end,2}   = iscell(eCell) && iscell(esCell) ...
                && isequal(size(eCell), [1, 3]);

[e_scalar, es_scalar] = edges(P_e(2, :)', [], 12);
results{end+1,1} = 'edges batched: matches scalar dispatch';
results{end,2}   = max(abs(eCell{2} - e_scalar)) < 1e-12 ...
                && max(abs(esCell{2} - es_scalar)) < 1e-12;

results{end+1,1} = 'edges batched: dedup row 1 = row 3';
results{end,2}   = isequal(eCell{1}, eCell{3}) && isequal(esCell{1}, esCell{3});

% --- projCentroid ---
P_pc = [0, 4, 7; 0, 3, 7; 0, 4, 7];
[yCell, cmCell, cpCell] = projCentroid(P_pc, [], 12);
results{end+1,1} = 'projCentroid batched: returns three 1-by-nRows cells';
results{end,2}   = iscell(yCell) && iscell(cmCell) && iscell(cpCell) ...
                && isequal(size(yCell), [1, 3]);

[y_scalar, cm_scalar, cp_scalar] = projCentroid(P_pc(2, :)', [], 12);
results{end+1,1} = 'projCentroid batched: matches scalar dispatch';
results{end,2}   = max(abs(yCell{2} - y_scalar)) < 1e-12 ...
                && abs(cmCell{2} - cm_scalar) < 1e-12 ...
                && abs(cpCell{2} - cp_scalar) < 1e-12;

results{end+1,1} = 'projCentroid batched: dedup centroid magnitude row 1 = row 3';
results{end,2}   = isequal(cmCell{1}, cmCell{3});

% --- circApm ---
P_apm = [0, 3, 6, 8, 10, 12, 14;
         0, 2, 4, 6, 8, 10, 12;
         0, 3, 6, 8, 10, 12, 14];
[Rcell, rPhaseCell, rLagCell] = circApm(P_apm, [], 16);
results{end+1,1} = 'circApm batched: returns three 1-by-nRows cells';
results{end,2}   = iscell(Rcell) && iscell(rPhaseCell) && iscell(rLagCell) ...
                && isequal(size(Rcell), [1, 3]) ...
                && isequal(size(Rcell{1}), [16, 16]);

% Scalar-equivalent: pre-sort to canonical form (batched dedups via
% sorted modular form before computing).
[R_scalar, ~, ~] = circApm(sort(mod(P_apm(2, :), 16))', [], 16);
results{end+1,1} = 'circApm batched: matches scalar dispatch (canonical form)';
results{end,2}   = isequal(Rcell{2}, R_scalar);

results{end+1,1} = 'circApm batched: dedup row 1 = row 3';
results{end,2}   = isequal(Rcell{1}, Rcell{3});

% Period reduction in canonical key
P_apm_pred = [0, 3, 6, 8, 10, 12, 14;
              0, 19, 6, 8, 10, 12, 14];   % 19 mod 16 = 3
[Rred, ~, ~] = circApm(P_apm_pred, [], 16);
results{end+1,1} = 'circApm batched: dedup over period reduction';
results{end,2}   = isequal(Rred{1}, Rred{2});

% Non-integer pitch errors clearly
results{end+1,1} = 'circApm batched: non-integer pitches error';
results{end,2}   = throwsErrorWithId( ...
    @() circApm([0.5, 1.0, 2.0; 0.5, 1.0, 2.0], [], 16), ...
    'circApm:nonIntegerPitch');


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
    fprintf('\n=== test_tier1_batched: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_tier1_batched:failed', '%d test(s) failed.', nFail);
    end
end
