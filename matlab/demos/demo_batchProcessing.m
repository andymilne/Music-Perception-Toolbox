%% demo_batchProcessing.m
%  Demonstrates batch computation of perceptual features on experimental
%  data with automatic deduplication of repeated weighted multisets.
%
%  Two deduplication workflows are shown:
%
%    1. Paired measures (SPCS) — use batchCosSimExpTens, which handles
%       deduplication internally.
%
%    2. Single-set measures (spectral entropy, template harmonicity,
%       tensor harmonicity, roughness) — use the unique/map pattern
%       shown below, which works for any function.
%
%  The dataset is synthetic: 3 scales × 4 chord types × 12 root
%  transpositions = 144 trials. Many trials share the same scale (3
%  unique) or the same chord pitch-class content (4 unique chord types,
%  regardless of transposition in the periodic case), so deduplication
%  avoids redundant computation.
%
%  Uses: batchCosSimExpTens, addSpectra, cosSimExpTens, spectralEntropy,
%        templateHarmonicity, tensorHarmonicity, roughness, convertPitch
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% Spectral parameters
nHarm  = 24;      % number of harmonics
rho    = 1;       % power-law rolloff exponent (1/n)
spec   = {'harmonic', nHarm, 'powerlaw', rho};

% Expectation tensor parameters
sigma  = 10;      % Gaussian smoothing width (cents)
r      = 1;       % monad expectation tensor
isRel  = 0;       % absolute (not transposition-invariant)
isPer  = 1;       % periodic (pitch-class equivalence)
period = 1200;    % one octave

% Reference pitch for roughness (Hz)
f0 = 261.63;      % middle C

%% === Create synthetic dataset ===

% Three 7-note scales (cents)
diatonic    = [0, 200, 400, 500, 700, 900, 1100];
harmonicMin = [0, 200, 300, 500, 700, 800, 1100];
melodicMin  = [0, 200, 300, 500, 700, 900, 1100];

scales     = [diatonic; harmonicMin; melodicMin];
scaleNames = {'Diatonic', 'Harmonic minor', 'Melodic minor'};

% Four chord types (cents relative to root)
chordTypes     = [0 400 700; 0 300 700; 0 300 600; 0 400 800];
chordTypeNames = {'Major', 'Minor', 'Dim', 'Aug'};

% 12 root pitch classes
roots  = 0:100:1100;
nRoots = numel(roots);

nScales = size(scales, 1);
nChords = size(chordTypes, 1);
nPairs  = nScales * nChords * nRoots;

% Build all (scale, transposed chord) pairs
pMatA = zeros(nPairs, 7);    % scales
pMatB = zeros(nPairs, 3);    % chords
scaleIdx = zeros(nPairs, 1);
chordIdx = zeros(nPairs, 1);
rootVals = zeros(nPairs, 1);

idx = 0;
for si = 1:nScales
    for ci = 1:nChords
        for ri = 1:nRoots
            idx = idx + 1;
            pMatA(idx, :) = scales(si, :);
            pMatB(idx, :) = chordTypes(ci, :) + roots(ri);
            scaleIdx(idx) = si;
            chordIdx(idx) = ci;
            rootVals(idx) = roots(ri);
        end
    end
end

fprintf('Dataset: %d trials (%d scales × %d chord types × %d roots).\n\n', ...
    nPairs, nScales, nChords, nRoots);

%% =====================================================================
%  WORKFLOW 1: Paired measure (SPCS) via batchCosSimExpTens
%  batchCosSimExpTens handles deduplication internally.
%  =====================================================================

fprintf('=== Workflow 1: SPCS via batchCosSimExpTens ===\n\n');

spcs = batchCosSimExpTens(pMatA, pMatB, ...
    sigma, r, isRel, isPer, period, ...
    'spectrum', spec);

spcs = round(spcs, 3);

% Display as scale × chord × root tables
for si = 1:nScales
    fprintf('\n  %s:\n', scaleNames{si});
    fprintf('  %-8s', '');
    for ri = 1:nRoots
        fprintf('%6d', roots(ri));
    end
    fprintf('\n');

    for ci = 1:nChords
        fprintf('  %-8s', chordTypeNames{ci});
        for ri = 1:nRoots
            mask = scaleIdx == si & chordIdx == ci & rootVals == roots(ri);
            fprintf('%6.3f', spcs(mask));
        end
        fprintf('\n');
    end
end

%% =====================================================================
%  WORKFLOW 2: Single-set measures via unique/map deduplication
%
%  The pattern is:
%    1. Sort each row and call unique(..., 'rows') to find unique sets
%    2. Compute the measure once per unique set
%    3. Map results back to all rows via the index from unique()
%
%  This works for any function. We demonstrate it here for spectral
%  entropy, template harmonicity, tensor harmonicity, and roughness
%  applied to the chords (pMatB).
%  =====================================================================

fprintf('\n=== Workflow 2: Single-set measures (chord features) ===\n');

% --- Step 1: Deduplicate ---
sortedB = sort(pMatB, 2);
[uniqueChords, ~, chordMap] = unique(sortedB, 'rows');
nUnique = size(uniqueChords, 1);

fprintf('\n  %d trials → %d unique chord multisets.\n\n', ...
    nPairs, nUnique);

% --- Step 2: Compute once per unique set ---
% Each function handles spectral enrichment via its own parameters,
% rather than pre-enriching all pitches (which would be prohibitively
% expensive for tensor harmonicity with many partials).
uSpecEnt  = NaN(nUnique, 1);
uHMax     = NaN(nUnique, 1);
uHEnt     = NaN(nUnique, 1);
uTensHarm = NaN(nUnique, 1);
uRough    = NaN(nUnique, 1);

refCents = convertPitch(f0, 'hz', 'cents');

for ui = 1:nUnique
    p = uniqueChords(ui, :);
    p = p(~isnan(p));  % strip NaN padding (if any)

    % Spectral entropy (uses 'spectrum' parameter internally)
    uSpecEnt(ui) = spectralEntropy(p(:), [], sigma, 'spectrum', spec);

    % Template harmonicity (uses 'chordSpectrum' parameter)
    [uHMax(ui), uHEnt(ui)] = templateHarmonicity(p(:), [], sigma, ...
        'chordSpectrum', spec);

    % Tensor harmonicity (uses 'spectrum' parameter for the template)
    uTensHarm(ui) = tensorHarmonicity(p(:), [], sigma, 'spectrum', spec);

    % Roughness (needs Hz and enriched spectra)
    [pSpec, wSpec] = addSpectra(p(:), [], spec{:});
    fHz = convertPitch(pSpec + refCents, 'cents', 'hz');
    uRough(ui) = roughness(fHz, wSpec);
end

% --- Step 3: Map back to all rows ---
specEnt  = uSpecEnt(chordMap);
hMax     = uHMax(chordMap);
hEnt     = uHEnt(chordMap);
tensHarm = uTensHarm(chordMap);
rough    = uRough(chordMap);

% --- Display ---
fprintf('  %-8s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Chord', 'specEnt', 'hMax', 'hEnt', 'tensHarm', 'Rough');
fprintf('  %s\n', repmat('-', 1, 56));

for ui = 1:nUnique
    % Find first trial with this unique chord to get the chord name
    firstIdx = find(chordMap == ui, 1);
    ci = chordIdx(firstIdx);
    ri = find(roots == rootVals(firstIdx));
    label = sprintf('%s @ %d', chordTypeNames{ci}, roots(ri));

    fprintf('  %-14s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        label, uSpecEnt(ui), uHMax(ui), uHEnt(ui), uTensHarm(ui), uRough(ui));
end

fprintf('\n  (Only %d unique computations needed instead of %d.)\n', ...
    nUnique, nPairs);

%% === Plot: SPCS heatmaps ===

figure('Name', 'Batch processing demo');
for si = 1:nScales
    subplot(1, nScales, si);

    S = NaN(nChords, nRoots);
    for ci = 1:nChords
        for ri = 1:nRoots
            mask = scaleIdx == si & chordIdx == ci & rootVals == roots(ri);
            S(ci, ri) = spcs(mask);
        end
    end

    imagesc(roots, 1:nChords, S);
    set(gca, 'YTick', 1:nChords, 'YTickLabel', chordTypeNames);
    xlabel('Root (cents)');
    title(scaleNames{si});
    colorbar;
end

sgtitle('SPCS: chord fit at each scale degree');
colormap(parula);

fprintf('\nDone.\n');
