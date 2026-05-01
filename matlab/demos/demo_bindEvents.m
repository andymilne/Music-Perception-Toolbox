%DEMO_BINDEVENTS Demonstrate bindEvents: sliding-window binding into
% n-attribute super-events.
%
% Combined with differenceEvents, bindEvents generalises the n-tuple
% entropy of Milne & Dean (2016) to non-zero sigma, continuous values,
% non-periodic domains, and per-event weights. More importantly, the
% bound MAET is a *density* that can be fed into the rest of the
% toolbox (cosSimExpTens, windowedSimilarity, etc.), rather than a
% scalar.
%
% Sections
%   1. n-tuple entropy via differenceEvents + bindEvents
%      (matches nTupleEntropy at sigma -> 0)
%   2. Smoothed extension (sigma > 0)
%   3. Cosine similarity between scales' n-tuple distributions
%   4. Melodic n-grams: binding a raw pitch sequence

clear; clc;

%% ===================================================================
%  1. n-tuple entropy via differenceEvents + bindEvents
%  ===================================================================

fprintf('=== 1. n-tuple entropy via differenceEvents + bindEvents ===\n');
fprintf('    (matches nTupleEntropy / Milne & Dean 2016 at sigma -> 0)\n\n');

p = [0 2 4 5 7 9 11];           % diatonic scale
P = 12;                          % 12-EDO

% Cyclic step sizes (K events -> K differences)
pSorted = sort(mod(p, P));
diffs = mod(diff([pSorted, pSorted(1) + P]), P);
fprintf('    Diatonic scale: [%s], period = %d\n', num2str(p), P);
fprintf('    Cyclic step sizes: [%s]\n\n', num2str(diffs));

fprintf('    %3s  %12s  %15s  %11s\n', 'n', 'bind+MAET', 'nTupleEntropy', 'difference');
for n = 1:3
    [pBound, wBound] = bindEvents(diffs, [], n, 'circular', true);
    sigma = 1e-6;
    T = buildExpTens(pBound, wBound, sigma, ones(1, n), ...
                     ones(1, n), false, true, P, ...
                     'verbose', false);
    H_bind = entropyExpTens(T, 'normalize', false, 'base', 2, ...
                             'nPointsPerDim', P);
    H_ref = nTupleEntropy(p, P, n, 'sigma', 0, 'normalize', false, ...
                           'base', 2);
    fprintf('    %3d  %12.6f  %15.6f  %+11.2e\n', ...
            n, H_bind, H_ref, H_bind - H_ref);
end


%% ===================================================================
%  2. Smoothed n-tuple entropy (sigma > 0)
%  ===================================================================

fprintf('\n=== 2. Smoothed n-tuple entropy (sigma > 0) ===\n');
fprintf('    Bind+MAET pipeline extends naturally to non-zero sigma.\n\n');

fprintf('    %3s  %6s  %12s  %15s\n', 'n', 'sigma', 'bind+MAET', 'nTupleEntropy');
for n = 1:3
    [pBound, wBound] = bindEvents(diffs, [], n, 'circular', true);
    for sigma = [0.5, 1.0]
        T = buildExpTens(pBound, wBound, sigma, ones(1, n), ...
                         ones(1, n), false, true, P, ...
                         'verbose', false);
        H_bind = entropyExpTens(T, 'normalize', false, 'base', 2, ...
                                 'nPointsPerDim', P);
        H_ref = nTupleEntropy(p, P, n, 'sigma', sigma, ...
                               'normalize', false, 'base', 2);
        fprintf('    %3d  %6.1f  %12.6f  %15.6f\n', ...
                n, sigma, H_bind, H_ref);
    end
end


%% ===================================================================
%  3. Cosine similarity between scales' n-tuple distributions
%  ===================================================================

fprintf('\n=== 3. Cosine similarity between scales'' n-tuple distributions ===\n');
fprintf('    Goes beyond the scalar nTupleEntropy: the bound MAET is a\n');
fprintf('    full density that can be compared to other densities.\n\n');

scaleNames = {'Major', 'Phrygian', 'Melodic minor (asc)', 'Harmonic minor'};
scales = {[0 2 4 5 7 9 11], ...
          [0 1 3 5 7 8 10], ...
          [0 2 3 5 7 9 11], ...
          [0 2 3 5 7 8 11]};

for k = 1:numel(scales)
    s = scales{k};
    sSorted = sort(mod(s, P));
    d = mod(diff([sSorted, sSorted(1) + P]), P);
    fprintf('    %-22s step sequence: [%s]\n', scaleNames{k}, num2str(d));
end
fprintf('\n');

sigma = 0.6;
fprintf('    Cosine similarity at sigma = %.1f:\n', sigma);
fprintf('    %-50s  %10s  %10s\n', '', 'n = 2', 'n = 3');

for i = 1:numel(scales)
    for j = i+1:numel(scales)
        row = sprintf('    sim(%-18s | %-18s)', scaleNames{i}, scaleNames{j});
        siSorted = sort(mod(scales{i}, P));
        di = mod(diff([siSorted, siSorted(1) + P]), P);
        sjSorted = sort(mod(scales{j}, P));
        dj = mod(diff([sjSorted, sjSorted(1) + P]), P);
        for n = [2, 3]
            [pBi, wBi] = bindEvents(di, [], n, 'circular', true);
            [pBj, wBj] = bindEvents(dj, [], n, 'circular', true);
            Ti = buildExpTens(pBi, wBi, sigma, ones(1, n), ...
                              ones(1, n), false, true, P, ...
                              'verbose', false);
            Tj = buildExpTens(pBj, wBj, sigma, ones(1, n), ...
                              ones(1, n), false, true, P, ...
                              'verbose', false);
            s = cosSimExpTens(Ti, Tj, 'verbose', false);
            row = [row, sprintf('  %10.4f', s)]; %#ok<AGROW>
        end
        fprintf('%s\n', row);
    end
end

fprintf('\n    Note: at n = 2 the diatonic-mode rotations all share a step-pair\n');
fprintf('    multiset, so they are indistinguishable. At n = 3 melodic minor\n');
fprintf('    separates from major / phrygian. Harmonic minor (with its\n');
fprintf('    augmented second) differs from the major modes at both n.\n\n');


%% ===================================================================
%  4. Melodic n-grams: binding raw pitches (no differencing)
%  ===================================================================

fprintf('=== 4. Melodic n-grams (binding raw pitches, no differencing) ===\n');
fprintf('    Skipping the difference step gives n-grams of pitches in their\n');
fprintf('    actual register, rather than transposition-invariant intervals.\n\n');

melody = [67 69 71 72 71 69 67 65 67];
fprintf('    Phrase (MIDI): [%s]\n', num2str(melody));

[pB, wB] = bindEvents(melody, [], 3, 'circular', false);
nPrime = size(pB{1}, 2);
fprintf('    bindEvents(n=3, circular=false) -> %d attributes of shape (1, %d):\n', ...
        numel(pB), nPrime);
for j = 1:numel(pB)
    fprintf('      lag %d: [%s]\n', j-1, num2str(pB{j}));
end

fprintf('\n    These three attributes, placed in one group with r = 1 each,\n');
fprintf('    form a 3-D MAET over (pitch_t, pitch_{t+1}, pitch_{t+2}).\n');
fprintf('    The resulting density characterises the melodic 3-gram\n');
fprintf('    structure of the phrase, in absolute pitch register.\n');
