%% test_cross_language_golden.m — v3 cross-language equivalence
%
%  Hardcodes outputs of representative v3 computations on
%  deterministic inputs (no RNG). The companion Python file
%  python/tests/test_cross_language_golden.py hardcodes the same
%  values; running both pins down cross-language numerical agreement
%  to 1e-8 relative on the v3 surface (Möbius cosine similarity single-multiset
%  + MA Rényi-2 entropy single-multiset + MA,
%  Möbius-method tensorHarmonicity, and Möbius-method evalExpTens).
%
%  Inputs use 'method', 'mobius' on the cosine cases so the Möbius method
%  Möbius machinery is genuinely exercised rather than the
%  dispatcher's cost-model fallback to Bulger. Sigmas are chosen
%  to keep values well-conditioned (away from FP underflow); a 1e-8
%  relative tolerance is the standard used elsewhere in the v22 suite.
%
%  When regenerating the golden table after a deliberate algorithm
%  change, update the values in BOTH this file and the Python mirror
%  to the same number of digits.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

RTOL = 1e-8;
ATOL = 1e-12;

%% ---- Case A: single-multiset cosSim, abs r=3, Möbius ----

p1 = [0; 400; 700];
p2 = [0; 300; 700];
w  = [1; 1; 1];
sA = cosSimExpTens(p1, w, p2, w, 80, 3, false, false, 0, ...
    'method', 'mobius', 'verbose', false);
GOLDEN_A = 0.67614851033133;
results{end+1, 1} = 'cross-language golden A: single-multiset cosSim abs r=3 Möbius';
results{end, 2}   = abs(sA - GOLDEN_A) < RTOL * abs(GOLDEN_A) + ATOL;

%% ---- Case B: single-multiset cosSim, rel r=3 per, Möbius ----

sB = cosSimExpTens(p1, w, p2, w, 80, 3, true, true, 1200, ...
    'method', 'mobius', 'verbose', false);
GOLDEN_B = 0.98878587398645;
results{end+1, 1} = 'cross-language golden B: single-multiset cosSim rel r=3 per Möbius';
results{end, 2}   = abs(sB - GOLDEN_B) < RTOL * abs(GOLDEN_B) + ATOL;

%% ---- Case C: MA cosSim ragged-K ----

P_x = [ 50  100  200  300  400  500;
       150  250  350  450  550  650;
       350  450  550  650  750  850;
       550  650  750  850  950 1050;
       NaN  850  950  NaN 1150 1250;
       NaN 1050 1150  NaN 1350 1450;
       NaN 1250 1350  NaN 1550 1650;
       NaN 1450 1550  NaN 1750 1850];   % (8, 6)
W_x = ones(size(P_x));
W_x(isnan(P_x)) = NaN;
P_y = P_x + 50;
W_y = W_x;
dx = buildExpTens({P_x}, {W_x}, 25, 3, false, false, 0, ...
    'verbose', false);
dy = buildExpTens({P_y}, {W_y}, 25, 3, false, false, 0, ...
    'verbose', false);
sC = cosSimExpTens(dx, dy, 'method', 'mobius', 'verbose', false);
GOLDEN_C = 0.12066345091832;
results{end+1, 1} = 'cross-language golden C: MA cosSim ragged-K';
results{end, 2}   = abs(sC - GOLDEN_C) < RTOL * abs(GOLDEN_C) + ATOL;

%% ---- Case D: single-multiset entropy Rényi-2, abs r=2 ----

HD = entropyExpTens(p1, w, 20, 2, false, false, 0, ...
    'method', 'renyi2', 'base', 2);
GOLDEN_D = 14.88031481996820;
results{end+1, 1} = 'cross-language golden D: single-multiset entropy Rényi-2 abs r=2';
results{end, 2}   = abs(HD - GOLDEN_D) < RTOL * abs(GOLDEN_D) + ATOL;

%% ---- Case E: MA entropy Rényi-2 ----

pitch = [   0   200   400   600;
          400   600   700   900;
          700   900  1000  1100;
         1000  1200  1300  1400;
         1100  1300  1500  1700];   % (5, 4)
time = [0 0.5 1.0 1.5];              % (1, 4)
HE = entropyExpTens({pitch, time}, [], ...
    [12, 0.05], [3, 1], ...
    [false, false], [true, false], [1200, 0], ...
    'method', 'renyi2', 'base', 2);
GOLDEN_E = 21.64284222436801;
results{end+1, 1} = 'cross-language golden E: MA entropy Rényi-2';
results{end, 2}   = abs(HE - GOLDEN_E) < RTOL * abs(GOLDEN_E) + ATOL;

%% ---- Case F: tensorHarmonicity Möbius method ----

hF = tensorHarmonicity([0; 400; 700], [], 12, 'verbose', false);
GOLDEN_F = 0.17358467740231;
results{end+1, 1} = 'cross-language golden F: tensorHarmonicity Möbius';
results{end, 2}   = abs(hF - GOLDEN_F) < RTOL * abs(GOLDEN_F) + ATOL;

%% ---- Case G: evalExpTens at single query, rel Möbius ----

tp = [0; 1200; 1902; 2400];
tw = [1; 0.5; 0.333; 0.25];
X  = [400; 700];   % 2 x 1 (r=3 rel -> dim=2)
v = evalExpTens(tp, tw, 80, 3, true, false, 1200, X, 'verbose', false);
GOLDEN_G = 2.07507623760499e-06;
results{end+1, 1} = 'cross-language golden G: evalExpTens rel orbit';
results{end, 2}   = abs(v - GOLDEN_G) < RTOL * abs(GOLDEN_G) + ATOL;

%% ---- Case H: single-multiset Shannon entropy abs r=2 dim=2 (bin-integration path) ----
% These cases lock in the bin-integration parity for the discrete entropy
% methods. The bin-integration path was added to Python without a
% parallel MATLAB port for a release window; these goldens catch any
% future drift between the per-axis Phi-difference contractions.

dens_h = buildExpTens([100; 200; 300], [], 20, 2, false, false, 0, ...
    'verbose', false);
HH = entropyExpTens(dens_h, 'method', 'shannon', ...
    'nPointsPerDim', 40, 'xMin', 50, 'xMax', 350, 'verbose', false);
GOLDEN_H = 9.383611317877847;
results{end+1, 1} = 'cross-language golden H: single-multiset shannon abs r=2 dim=2 bin-integration';
results{end, 2}   = abs(HH - GOLDEN_H) < RTOL * abs(GOLDEN_H) + ATOL;

%% ---- Case I: single-multiset normalized (Pielou ratio) ----

HI = entropyExpTens(dens_h, 'method', 'normalized', ...
    'nPointsPerDim', 40, 'xMin', 50, 'xMax', 350, 'verbose', false);
GOLDEN_I = 0.8815988444951405;
results{end+1, 1} = 'cross-language golden I: single-multiset normalized abs r=2 dim=2';
results{end, 2}   = abs(HI - GOLDEN_I) < RTOL * abs(GOLDEN_I) + ATOL;

%% ---- Case J: single-multiset Shannon periodic r=1 ----

dens_j = buildExpTens([0; 3; 7], [], 0.7, 1, false, true, 12, ...
    'verbose', false);
HJ = entropyExpTens(dens_j, 'method', 'shannon', ...
    'nPointsPerDim', 24, 'verbose', false);
GOLDEN_J = 4.093676510565166;
results{end+1, 1} = 'cross-language golden J: single-multiset shannon periodic r=1';
results{end, 2}   = abs(HJ - GOLDEN_J) < RTOL * abs(GOLDEN_J) + ATOL;

%% ---- Case K: MA Shannon abs dim=2 ----

P2 = [100, 200, 300; 200, 250, 100];
W2 = [1, 1, 1];
dens_k = buildExpTens({P2}, {W2}, 20, 2, false, false, 0, ...
    'verbose', false);
HK = entropyExpTens(dens_k, 'method', 'shannon', ...
    'nPointsPerDim', 40, 'xMin', 50, 'xMax', 350, 'verbose', false);
GOLDEN_K = 9.347143263809102;
results{end+1, 1} = 'cross-language golden K: MA shannon abs dim=2';
results{end, 2}   = abs(HK - GOLDEN_K) < RTOL * abs(GOLDEN_K) + ATOL;

%% ---- Case L: single-multiset differential entropy (adaptive) ----
% Adaptive convergence tolerance is ~exp(-18) ~ 1.5e-8;
% allow 1e-5 absolute as a comfortable bound.

dens_l = buildExpTens([0; 400; 700], [], 20, 1, false, false, 0, ...
    'verbose', false);
HL = entropyExpTens(dens_l, 'method', 'differential', 'verbose', false);
GOLDEN_L = 7.953986161000217;
results{end+1, 1} = 'cross-language golden L: single-multiset differential r=1 adaptive';
results{end, 2}   = abs(HL - GOLDEN_L) < 1e-5;

%% ---- Case M: MA differential entropy (adaptive, D==2) ----
% Two attributes, r=1 each (D==2). Exercises the D==2 leading-axis
% cell-block streaming in localContractCellAxes. Narrow spans and an
% explicit truncationSigmas keep the converged grid small, so the value
% is reproducible at the looser adaptive tolerance.

P0_m = [0, 120, 260];
P1_m = [0, 80, 170];
W0_m = [1.0, 0.7, 0.5];
W1_m = [1.0, 1.0, 1.0];
HM = entropyExpTens({P0_m, P1_m}, {W0_m, W1_m}, [45, 35], [1, 1], ...
    [false, false], [false, false], [0, 0], ...
    'method', 'differential', 'base', 2, ...
    'truncationSigmas', 3.0, 'verbose', false);
GOLDEN_M = 16.07500552585256;
results{end+1, 1} = 'cross-language golden M: MA differential D==2 adaptive';
results{end, 2}   = abs(HM - GOLDEN_M) < 1e-4;

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = numel(results(:, 1)) - nPass;
    fprintf('\n=== test_cross_language_golden: %d passed, %d failed (of %d) ===\n', ...
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
