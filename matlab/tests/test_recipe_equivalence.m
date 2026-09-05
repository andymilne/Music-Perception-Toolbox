%% test_recipe_equivalence.m — recipe path agrees with contract on the same inputs
%
%  Targeted regression tests confirming that
%    out_recipe = mobius.executeRecipe(operands, mobius.buildContractRecipe(opAxes, freeAxes))
%  agrees with
%    out_contract = reference.contract(operands, opAxes, freeAxes)
%  to floating-point precision across the call patterns used by
%  the orbit-IP consumers (mobius.innerProductOrbit,
%  mobius.innerProductOrbitGrid, mobius.innerProductOrbitPwBatched).
%
%  The recipe path is structurally the v2.2 fast path: contraction
%  graph is computed once at orbit-table-build time, runtime executes
%  precomputed permutations and reshapes. The dynamic-dispatch
%  contract path remains as a reference / fallback. Both must agree.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    addpath(fullfile(fileparts(mfilename('fullpath')), 'reference'));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

ATOL = 1e-12;
RTOL = 1e-12;

%% ---- Simple matmul ----

rng(11, 'twister');
A = randn(4, 5);
B = randn(5, 6);
opAxes = {[1 2], [2 3]};
freeAxes = [1 3];
operands = {A, B};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: matmul (2 ops, free=2 axes)';
results{end, 2}   = max(abs(out_c(:) - out_r(:))) < ATOL + RTOL * max(abs(out_c(:)));

%% ---- Trace (full contraction, scalar output) ----

rng(13, 'twister');
A = randn(5, 5);
B = randn(5, 5);
opAxes = {[1 2], [2 1]};
freeAxes = [];
operands = {A, B};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: trace (full contract)';
results{end, 2}   = abs(out_c - out_r) < ATOL + RTOL * abs(out_c);

%% ---- Bilinear form u' K v ----

rng(17, 'twister');
u = randn(5, 1);
K = randn(5, 4);
v = randn(4, 1);
opAxes = {1, [1 2], 2};
freeAxes = [];
operands = {u, K, v};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: bilinear u'' K v';
results{end, 2}   = abs(out_c - out_r) < ATOL + RTOL * abs(out_c);

%% ---- Shared axis across 3 operands ----

rng(19, 'twister');
A = randn(4, 5);
B = randn(4, 6);
u = randn(4, 1);
opAxes = {[1 2], [1 3], 1};
freeAxes = [];
operands = {A, B, u};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: 3-operand shared axis';
results{end, 2}   = abs(out_c - out_r) < ATOL + RTOL * abs(out_c);

%% ---- Orbit-like at r=2 ----

rng(23, 'twister');
n_A = 6; n_B = 6;
w_A = 0.5 + rand(n_A, 1);
w_B = 0.5 + rand(n_B, 1);
K = exp(-rand(n_A, n_B));   % positive kernel
% r=2 single-block orbit: 2 weight powers, 1 kernel power, 4 axes total
opAxes = {1, 2, [1 2]};   % qA=1, qB=1, 1 edge
freeAxes = [];
operands = {w_A.^2, w_B.^2, K.^1};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: orbit-like r=2 single-block';
results{end, 2}   = abs(out_c - out_r) < ATOL + RTOL * abs(out_c);

%% ---- Orbit-like at r=3 (more operands) ----

rng(29, 'twister');
n_A = 8; n_B = 8;
w_A = 0.5 + rand(n_A, 1);
w_B = 0.5 + rand(n_B, 1);
K = exp(-rand(n_A, n_B));
% r=3 with qA=2, qB=1 partition, edges [(1,1,2),(2,1,1)]
opAxes = {1, 2, 3, [1 3], [2 3]};
freeAxes = [];
operands = {w_A.^2, w_A.^1, w_B.^3, K.^2, K.^1};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: orbit-like r=3 (qA=2, qB=1)';
results{end, 2}   = abs(out_c - out_r) < ATOL + RTOL * abs(out_c);

%% ---- Free axis carried through (Grid pattern) ----

rng(31, 'twister');
N_u = 10;
n_A = 5; n_B = 5;
U_LABEL = 1000;
w_A = 0.5 + rand(n_A, 1);
w_B = 0.5 + rand(n_B, 1);
K_u = exp(-rand(N_u, n_A, n_B));
opAxes = {1, 2, [U_LABEL, 1, 2]};
freeAxes = U_LABEL;
operands = {w_A, w_B, K_u};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: Grid-pattern (U on kernels only)';
results{end, 2}   = max(abs(out_c(:) - out_r(:))) < ATOL + RTOL * max(abs(out_c(:)));

%% ---- Free axis on every operand (PwBatched pattern) ----

rng(37, 'twister');
N = 8;
n_A = 5; n_B = 5;
w_A_g = 0.5 + rand(N, n_A);
w_B_g = 0.5 + rand(N, n_B);
K_g = exp(-rand(N, n_A, n_B));
opAxes = {[U_LABEL, 1], [U_LABEL, 2], [U_LABEL, 1, 2]};
freeAxes = U_LABEL;
operands = {w_A_g, w_B_g, K_g};
out_c = reference.contract(operands, opAxes, freeAxes);
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
out_r = mobius.executeRecipe(operands, recipe);
results{end+1, 1} = 'recipe equiv: PwBatched-pattern (U on every operand)';
results{end, 2}   = max(abs(out_c(:) - out_r(:))) < ATOL + RTOL * max(abs(out_c(:)));

%% ---- Recipe is reusable across calls with different operand sizes ----
% Build recipe once, execute with operands of varying sizes — checks that
% the recipe's permutations don't bake in any size assumptions.

opAxes = {1, 2, [1 2]};   % bilinear, full contract
freeAxes = [];
recipe = mobius.buildContractRecipe(opAxes, freeAxes);
allOK = true;
for trial = 1:5
    rng(100 + trial, 'twister');
    n_A = 3 + trial; n_B = 4 + trial;
    u = randn(n_A, 1);
    K = randn(n_A, n_B);
    v = randn(n_B, 1);
    operands = {u, v, K};
    out_c = reference.contract(operands, opAxes, freeAxes);
    out_r = mobius.executeRecipe(operands, recipe);
    if abs(out_c - out_r) > ATOL + RTOL * abs(out_c)
        allOK = false;
        break;
    end
end
results{end+1, 1} = 'recipe equiv: reusable across operand sizes';
results{end, 2}   = allOK;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = numel(results(:, 1)) - nPass;
    fprintf('\n=== test_recipe_equivalence: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:, 1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
