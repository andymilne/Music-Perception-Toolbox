function tests = test_wrapper_routing
%TEST_WRAPPER_ROUTING  Stage 2c invariants for user-facing wrappers.
%
%   Verifies that tensorHarmonicity, templateHarmonicity, virtualPitches,
%   and spectralEntropy:
%     (a) accept 'truncationSigmas' and 'kernelPrecision' name-value pairs
%         without error;
%     (b) return values that match between default (exact) and
%         truncationSigmas=6 modes to within the truncation tolerance;
%     (c) respect the global mptDefaults setting for these kwargs.
%
%   Stage 2c is the work that rerouted tensorHarmonicity through
%   evalExpTens (removing a wrapper-level algorithm-choice violation)
%   and threaded truncationSigmas / kernelPrecision through all four
%   wrappers so that the v2.2 helper-accelerated centres path is reached
%   from any user-facing entry point.

    tests = functiontests(localfunctions);
end


% =========================================================================
%  Chord battery
% =========================================================================

function chords = makeChordBattery()
    chords = struct();
    chords.unison           = [0, 1];
    chords.majorTriad       = [0, 400, 700];
    chords.minorTriad       = [0, 300, 700];
    chords.diminishedTriad  = [0, 300, 600];
    chords.augmentedTriad   = [0, 400, 800];
    chords.narrowCluster    = [0, 50, 100];
    chords.octavePlusFifth  = [0, 700, 1200];
end


% =========================================================================
%  tensorHarmonicity: the wrapper that was rerouted in Stage 2c
% =========================================================================

function test_tensorHarmonicity_scalar_exact_vs_truncated(testCase)
    mptDefaults('reset');
    chords = makeChordBattery();
    fns = fieldnames(chords);
    for k = 1:numel(fns)
        chord = chords.(fns{k});
        h_exact = tensorHarmonicity(chord, [], 12, 'verbose', false);
        h_trunc = tensorHarmonicity(chord, [], 12, ...
            'truncationSigmas', 6, 'verbose', false);
        verifyLessThan(testCase, abs(h_exact - h_trunc), 1e-7, ...
            sprintf('%s: exact %.6f, trunc %.6f, diff %.2e', ...
                fns{k}, h_exact, h_trunc, abs(h_exact - h_trunc)));
    end
end

function test_tensorHarmonicity_global_default_propagates(testCase)
    mptDefaults('reset');
    chord = [0, 400, 700];
    h_explicit = tensorHarmonicity(chord, [], 12, ...
        'truncationSigmas', 6, 'verbose', false);
    mptDefaults('truncationSigmas', 6);
    cleanupObj = onCleanup(@() mptDefaults('reset'));
    h_global = tensorHarmonicity(chord, [], 12, 'verbose', false);
    verifyEqual(testCase, h_explicit, h_global, ...
        'Global default truncationSigmas not propagating into wrapper');
end

function test_tensorHarmonicity_batched_matches_scalar(testCase)
    mptDefaults('reset');
    P = [0, 400, 700;
         0, 300, 700;
         0, 300, 600];
    H_batch = tensorHarmonicity(P, [], 12, ...
        'truncationSigmas', 6, 'verbose', false);
    H_scalar = zeros(size(P, 1), 1);
    for i = 1:size(P, 1)
        H_scalar(i) = tensorHarmonicity(P(i, :), [], 12, ...
            'truncationSigmas', 6, 'verbose', false);
    end
    verifyEqual(testCase, H_batch, H_scalar, 'AbsTol', 1e-12);
end


% =========================================================================
%  templateHarmonicity, virtualPitches, spectralEntropy:
%  already route correctly; verify kwargs threading.
% =========================================================================

function test_templateHarmonicity_accepts_truncation_kwargs(testCase)
    mptDefaults('reset');
    chord = [0, 400, 700];
    [h_max_exact, ~] = templateHarmonicity(chord, [], 12, 'verbose', false);
    [h_max_trunc, ~] = templateHarmonicity(chord, [], 12, ...
        'truncationSigmas', 6, 'verbose', false);
    verifyLessThan(testCase, abs(h_max_exact - h_max_trunc), 1e-5);
end

function test_virtualPitches_accepts_truncation_kwargs(testCase)
    mptDefaults('reset');
    chord = [0, 400, 700];
    [vp_p_exact, vp_w_exact] = virtualPitches(chord, [], 12, 'verbose', false);
    [vp_p_trunc, vp_w_trunc] = virtualPitches(chord, [], 12, ...
        'truncationSigmas', 6, 'verbose', false);
    verifyEqual(testCase, vp_p_exact, vp_p_trunc, 'AbsTol', 1e-12);
    verifyEqual(testCase, vp_w_exact, vp_w_trunc, 'AbsTol', 1e-5);
end

function test_spectralEntropy_accepts_truncation_kwargs(testCase)
    mptDefaults('reset');
    chord = [0, 400, 700];
    spec = {'harmonic', 12, 'powerlaw', 1};
    h_exact = spectralEntropy(chord, [], 12, ...
        'spectrum', spec, 'verbose', false);
    h_trunc = spectralEntropy(chord, [], 12, ...
        'spectrum', spec, 'truncationSigmas', 6, 'verbose', false);
    verifyLessThan(testCase, abs(h_exact - h_trunc), 1e-5);
end
