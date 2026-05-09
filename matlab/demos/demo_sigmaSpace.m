%% demo_sigmaSpace.m
%  Demonstration of the v2.1.0 sigma + sigmaSpace additions to
%  sameness, coherence, and nTupleEntropy.
%
%  These three functions previously had no soft (sigma > 0) versions
%  in v2.0; v2.1 introduces a Gaussian-kernel-based soft path with a
%  flag controlling how sigma is interpreted:
%
%    sigmaSpace = 'position' (default)
%       sigma is positional uncertainty on each input value p_k.
%       Cross-event correlations between intervals derived from a
%       shared position set are captured exactly via per-pair
%       variance bookkeeping.
%
%    sigmaSpace = 'interval'
%       sigma is independent uncertainty per derived interval.
%       Slots are treated as independent draws (V = 2 sigma^2
%       uniformly).
%
%  The two flags coincide at sigma = 0; only at sigma > 0 does the
%  distinction matter.

clear; close all;

DIATONIC = [0, 2, 4, 5, 7, 9, 11];
PERIOD = 12;
SIGMAS = [0, 0.1, 0.25, 0.5, 1.0, 2.0];

%% ===== 1. Sameness =====

fprintf('\n=== Sameness on the diatonic scale [0,2,4,5,7,9,11] in 12-EDO ===\n\n');
fprintf('  At sigma = 0 the v2.0 hard count is recovered exactly:\n');
fprintf('    one ambiguity (the tritone, size 6 as both 4th and 5th).\n\n');

fprintf('  %-10s %-15s %-15s\n', 'sigma', 'sq (position)', 'sq (interval)');
fprintf('  %s\n', repmat('-', 1, 42));
for s = SIGMAS
    if s == 0
        sq = sameness(DIATONIC, PERIOD, 0);
        sqP = sq; sqI = sq;
    else
        sqP = sameness(DIATONIC, PERIOD, s, 'sigmaSpace', 'position');
        sqI = sameness(DIATONIC, PERIOD, s, 'sigmaSpace', 'interval');
    end
    fprintf('  %-10g %-15.4f %-15.4f\n', s, sqP, sqI);
end
fprintf('\n  Position is more aggressive (lower sq) at typical sigma\n');
fprintf('  because its per-pair variance for disjoint-endpoint pairs\n');
fprintf('  is 4*sigma^2 — wider than interval''s uniform 2*sigma^2,\n');
fprintf('  so the soft-match kernel is broader.\n');

%% ===== 2. Coherence =====

fprintf('\n=== Coherence on the diatonic scale ===\n\n');
fprintf('  v2.0 strict coherence: 1 failure (the tritone), c = 1 - 1/140.\n');
fprintf('  Soft sigma -> 0+ limit:   tritone counts as 0.5 of a failure\n');
fprintf('  (the means D2 = D1 give P(D2 <= D1) = 0.5 exactly under any\n');
fprintf('  sigma > 0), so c -> 1 - 0.5/140 = %.4f.\n\n', 1 - 0.5/140);

fprintf('  %-10s %-15s %-15s\n', 'sigma', 'c (position)', 'c (interval)');
fprintf('  %s\n', repmat('-', 1, 42));
for s = SIGMAS
    if s == 0
        c = coherence(DIATONIC, PERIOD, 0);
        cP = c; cI = c;
    else
        cP = coherence(DIATONIC, PERIOD, s, 'sigmaSpace', 'position');
        cI = coherence(DIATONIC, PERIOD, s, 'sigmaSpace', 'interval');
    end
    fprintf('  %-10g %-15.4f %-15.4f\n', s, cP, cI);
end
fprintf('\n  The discontinuity at sigma = 0 (0.9929 to 0.9964 between\n');
fprintf('  the v2.0 hard count and the soft path''s sigma -> 0+ limit)\n');
fprintf('  is intentional: the strict flag splits ties as you choose at\n');
fprintf('  sigma exactly zero, while the soft path averages over them.\n');

%% ===== 3. The tritone diagnostic =====

fprintf('\n=== The tritone tie under positional jitter ===\n\n');
fprintf('  In the diatonic scale, the fourth F->B (positions 5->11)\n');
fprintf('  and the fifth B->F (positions 11->5, wrapping) share both\n');
fprintf('  endpoints. Under positional jitter on F and B, the\n');
fprintf('  difference D2 - D1 has variance 8*sigma^2 (not 4*sigma^2:\n');
fprintf('  the shared positions contribute with reinforcing signs).\n\n');
fprintf('  Both intervals have specific size = 6, so D2 - D1 has\n');
fprintf('  mean 0. P(D2 <= D1) = Phi(0) = 0.5, regardless of sigma.\n\n');

V = positionVariance([3+1, 6+1, 6+1, 3+1], [+1, -1, -1, +1], 1);
fprintf('  positionVariance for the tritone pair (sigma=1): V = %g\n', V);
fprintf('  (matches expected 8 for the shared-reinforcing case)\n');

%% ===== 4. nTupleEntropy =====

fprintf('\n=== nTupleEntropy on the diatonic ===\n\n');
fprintf('  At n = 1, sigma_space = ''position'' is exactly equivalent\n');
fprintf('  to sigma_space = ''interval'' with sigma multiplied by\n');
fprintf('  sqrt(2) — the marginal-matched relationship that captures\n');
fprintf('  the variance that one positional jitter induces in\n');
fprintf('  the derived step sizes.\n\n');

fprintf('  n = 1:\n');
fprintf('  %-10s %-18s %-18s\n', 'sigma', 'H (position)', 'H (interval)');
fprintf('  %s\n', repmat('-', 1, 48));
for s = [0, 0.1, 0.25, 0.5]
    if s == 0
        H = nTupleEntropy(DIATONIC, PERIOD, 1);
        Hp = H; Hi = H;
    else
        Hp = nTupleEntropy(DIATONIC, PERIOD, 1, 'sigma', s, ...
                           'sigmaSpace', 'position');
        Hi = nTupleEntropy(DIATONIC, PERIOD, 1, 'sigma', s, ...
                           'sigmaSpace', 'interval');
    end
    fprintf('  %-10g %-18.6f %-18.6f\n', s, Hp, Hi);
end

fprintf('\n  Verifying the n=1 exactness relationship:\n');
sigmaP = 0.5;
Hp = nTupleEntropy(DIATONIC, PERIOD, 1, 'sigma', sigmaP, ...
                   'sigmaSpace', 'position');
Hi_match = nTupleEntropy(DIATONIC, PERIOD, 1, 'sigma', sigmaP*sqrt(2), ...
                         'sigmaSpace', 'interval');
fprintf('    H(sigma=%.4f, position) = %.10f\n', sigmaP, Hp);
fprintf('    H(sigma=%.4f, interval) = %.10f\n', sigmaP*sqrt(2), Hi_match);
fprintf('    Difference: %.2e (should be ~zero)\n', abs(Hp - Hi_match));

fprintf('\n  At n >= 2, sigma_space = ''position'' falls back to a\n');
fprintf('  marginal-matched approximation that ignores anti-\n');
fprintf('  correlations between adjacent step slots. A warning\n');
fprintf('  fires once per call to make this explicit:\n\n');

origState = warning('on', 'nTupleEntropy:positionApprox');
H_n2_pos = nTupleEntropy(DIATONIC, PERIOD, 2, 'sigma', 0.3, ...
                         'sigmaSpace', 'position');
warning(origState);
fprintf('    H(diatonic, n=2, sigma=0.3, position) = %.6f\n', H_n2_pos);

fprintf('\nDone.\n');
