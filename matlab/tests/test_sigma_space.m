%% test_sigma_space.m — sigmaSpace position-aware soft measures (v3)
%
%  Tests for sigmaSpace position-aware soft measures (v3).
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
%  Tests for the sigma + sigmaSpace additions to sameness,
%  coherence, and nTupleEntropy (v3). Insert after the existing
%  Circular measures and Entropy sections.

% --- internal.positionVariance helper: signed-coefficient cases --------------

V = internal.positionVariance([1, 2, 3, 4], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: disjoint endpoints -> 4 sigma^2';
results{end,2}   = abs(V - 4) < 1e-12;

V = internal.positionVariance([1, 2, 1, 3], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: shared cancelling -> 2 sigma^2';
results{end,2}   = abs(V - 2) < 1e-12;

V = internal.positionVariance([2, 1, 1, 2], [+1, -1, -1, +1], 1.0);
results{end+1,1} = 'positionVariance: shared reinforcing -> 8 sigma^2';
results{end,2}   = abs(V - 8) < 1e-12;

V = internal.positionVariance([1, 2], [+1, -1], 0.5);
results{end+1,1} = 'positionVariance: scales with sigma^2';
results{end,2}   = abs(V - 0.5) < 1e-12;

% --- sameness: sigma = 0 byte-equivalence with v2 -------------------

[sq0, nd0] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0);
[sq1, nd1] = sameness([0, 2, 4, 5, 7, 9, 11], 12);     % default sigma = 0
results{end+1,1} = 'sameness: sigma=0 vs default agree (diatonic)';
results{end,2}   = sq0 == sq1 && nd0 == nd1;

[sqW0, ndW0] = sameness([0, 2, 4, 6, 8, 10], 12, 0);
results{end+1,1} = 'sameness: sigma=0 whole-tone perfect';
results{end,2}   = abs(sqW0 - 1) < 1e-12 && ndW0 == 0;

% --- sameness: sigma=0 flags coincide -------------------------------

[sqP, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0, 'sigmaSpace', 'position');
[sqI, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0, 'sigmaSpace', 'interval');
results{end+1,1} = 'sameness: sigma=0 flags coincide';
results{end,2}   = abs(sqP - sqI) < 1e-12;

% --- sameness: sigma>0 numerical regression -------------------------
%
%  Reference numbers from the Python verification of the same
%  implementation (computed at the same sigma values) — see
%  the v3 design discussion. These pin down the soft-path
%  computation against future regressions.

[sqP, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'position');
[sqI, ~] = sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'interval');
results{end+1,1} = 'sameness: diatonic sigma=0.5 position ~ 0.4224';
results{end,2}   = abs(sqP - 0.422420) < 1e-4;
results{end+1,1} = 'sameness: diatonic sigma=0.5 interval ~ 0.7144';
results{end,2}   = abs(sqI - 0.714369) < 1e-4;
results{end+1,1} = 'sameness: position more aggressive than interval';
results{end,2}   = sqP < sqI;

% --- sameness: float positions accepted when sigma > 0 --------------

ji = [0, 203.91, 386.31, 498.04, 701.96, 884.36, 1088.27];
[sqJI, ~] = sameness(ji, 1200, 25);
results{end+1,1} = 'sameness: JI diatonic accepts float p (sigma>0)';
results{end,2}   = isfinite(sqJI) && sqJI > 0 && sqJI <= 1.0;

% --- sameness: integer required at sigma = 0 ------------------------

results{end+1,1} = 'sameness: float p errors at sigma=0';
results{end,2}   = throwsErrorWithId( ...
    @() sameness([0.5, 2, 4, 7], 12, 0), ...
    'sameness:nonIntegerPositions');

% --- sameness: invalid sigmaSpace errors ----------------------------

results{end+1,1} = 'sameness: invalid sigmaSpace errors';
results{end,2}   = throwsError( ...
    @() sameness([0, 2, 4, 5, 7, 9, 11], 12, 0.5, 'sigmaSpace', 'bogus'));

% --- coherence: sigma = 0 byte-equivalence with v2 ------------------

[c0, nc0] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0);
[c1, nc1] = coherence([0, 2, 4, 5, 7, 9, 11], 12);   % default sigma = 0
results{end+1,1} = 'coherence: sigma=0 vs default agree';
results{end,2}   = c0 == c1 && nc0 == nc1;

[c0, nc0] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0, 'strict', false);
results{end+1,1} = 'coherence: sigma=0 strict=false (diatonic) -> nc=0';
results{end,2}   = nc0 == 0 && abs(c0 - 1.0) < 1e-12;

% --- coherence: sigma>0 numerical regression -------------------------

[cP, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'position');
[cI, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, ...
                    'sigmaSpace', 'interval');
results{end+1,1} = 'coherence: diatonic sigma=0.5 position ~ 0.8735';
results{end,2}   = abs(cP - 0.873485) < 1e-4;
results{end+1,1} = 'coherence: diatonic sigma=0.5 interval ~ 0.9446';
results{end,2}   = abs(cI - 0.944610) < 1e-4;

% --- coherence: tritone gives 0.5 contribution under position -------
%
%  The diatonic tritone is a fourth (F-B) and a fifth (B-F) sharing
%  endpoints with reinforcing signs. Var(D2 - D1) = 8 sigma^2 and
%  the means coincide exactly (both = 6 chromatic steps), so
%  P(D2 <= D1) = 0.5 at every sigma > 0. The soft-path nc as
%  sigma -> 0 therefore approaches 0.5 (one tritone, contributing
%  half a failure), giving c -> 1 - 0.5/140 = 0.99643.
[cLim, ~] = coherence([0, 2, 4, 5, 7, 9, 11], 12, 1e-6);
results{end+1,1} = 'coherence: sigma->0+ limit handles tritone tie';
results{end,2}   = abs(cLim - (1 - 0.5/140)) < 1e-3;

% --- coherence: invalid sigmaSpace errors ---------------------------

results{end+1,1} = 'coherence: invalid sigmaSpace errors';
results{end,2}   = throwsError( ...
    @() coherence([0, 2, 4, 5, 7, 9, 11], 12, 0.5, 'sigmaSpace', 'bogus'));

% --- nTupleEntropy: sigma=0 byte-equivalence with v2 ----------------

H0a = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12);
H0b = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0);
results{end+1,1} = 'nTupleEntropy: sigma=0 equals default';
results{end,2}   = abs(H0a - H0b) < 1e-12;

% --- nTupleEntropy: sigma=0 position == interval (published anchor) ---
%
%  At sigma = 0 both modes reduce to the integer step histogram (the
%  published Milne & Dean value), so position and interval coincide.

Hpos0 = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                      'sigma', 0, 'sigmaSpace', 'position', 'method', 'shannon');
Hint0 = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                      'sigma', 0, 'sigmaSpace', 'interval', 'method', 'shannon');
results{end+1,1} = 'nTupleEntropy: sigma=0 position = interval (n=2)';
results{end,2}   = abs(Hpos0 - Hint0) < 1e-12;

% --- nTupleEntropy: smoothing increases entropy ---------------------

H_raw    = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1);
H_smooth = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 1, 'sigma', 0.2);
results{end+1,1} = 'nTupleEntropy: smoothing increases H (n=1, position)';
results{end,2}   = H_smooth > H_raw;

% --- nTupleEntropy: position differs from interval at sigma > 0 -----

Hpos = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                     'sigma', 0.3, 'sigmaSpace', 'position');
Hint = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                     'sigma', 0.3, 'sigmaSpace', 'interval');
results{end+1,1} = 'nTupleEntropy: position differs from interval at sigma>0';
results{end,2}   = abs(Hpos - Hint) > 1e-6;

% --- nTupleEntropy: float positions accepted when sigma > 0 ---------

H_ji = nTupleEntropy(ji, 1200, 1, 'sigma', 25);
results{end+1,1} = 'nTupleEntropy: JI diatonic accepts float p (sigma>0)';
results{end,2}   = isfinite(H_ji) && H_ji > 0;

% --- nTupleEntropy: integer required at sigma = 0 -------------------

results{end+1,1} = 'nTupleEntropy: float p errors at sigma=0';
results{end,2}   = throwsErrorWithId( ...
    @() nTupleEntropy([0.5, 2, 4, 5, 7, 9, 11], 12, 1), ...
    'nTupleEntropy:nonIntegerPositions');

% --- nTupleEntropy: n>=2 position is exact (no approximation warning) ---
%
%  Position mode is now the exact correlated model at all n, so it
%  must not emit the old marginal-matched approximation warning.

lastwarn('');
H_n2_pos = nTupleEntropy([0, 2, 4, 5, 7, 9, 11], 12, 2, ...
                         'sigma', 0.3, 'sigmaSpace', 'position');
[~, warnId] = lastwarn;
results{end+1,1} = 'nTupleEntropy: n=2 position computes, no approx warning';
results{end,2}   = isfinite(H_n2_pos) && H_n2_pos > 0 ...
                   && ~strcmp(warnId, 'nTupleEntropy:positionApprox');


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
    fprintf('\n=== test_sigma_space: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_sigma_space:failed', '%d test(s) failed.', nFail);
    end
end
