%% demo_triadSpcsGrid.m
%  Spectral pitch class similarity (SPCS) of all possible 12-EDO triads
%  containing a perfect fifth, relative to a user-specified reference triad.
%
%  An example of this type of plot appears as Figure 3 in:
%    Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
%    Modelling the similarity of pitch collections with expectation tensors.
%    Journal of Mathematics and Music, 5(1), 1-20.
%
%  Each comparison triad has two degrees of freedom:
%    - The root of the fifth (which determines the fifth = root + 700)
%    - The position of the remaining note (the "third")
%
%  These two parameters form the axes of a 12x12 grid, centred on the
%  reference triad. The colour at each grid point indicates the SPCS with
%  the reference triad. The Euclidean distance between any two grid points
%  equals the voice-leading distance between the corresponding triads.
%
%  Uses: batchCosSimExpTens, addSpectra, cosSimExpTens
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% Reference triad (in cents). The plot is centred on this chord.
%   [0, 400, 700] = C major
%   [0, 300, 700] = C minor
refPitches = [0, 400, 700];
refName    = 'C major';

% Spectral parameters
nHarm  = 24;      % number of harmonics
rho    = 1;       % power-law rolloff exponent (1/n)

% Expectation tensor parameters
sigma  = 10;      % Gaussian smoothing width in cents
r      = 1;       % monad expectation tensor
isRel  = 0;       % absolute (not transposition-invariant)
isPer  = 1;       % periodic (pitch-class equivalence)
period = 1200;    % one octave in cents

% Display options
showLabels = true;  % set to false to hide chord labels

%% === Fixed definitions ===

% 12-EDO pitch classes (cents)
pcs = 0:100:1100;
nPCs = numel(pcs);

% Note names
noteNames = {'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'};

refRoot  = refPitches(1);
refThird = refPitches(2);

%% === Compute similarities ===

% Build all 144 comparison triads: {root, third, root + 700}
[rootGrid, thirdGrid] = meshgrid(pcs, pcs);
pMatB = [rootGrid(:), thirdGrid(:), rootGrid(:) + 700];

% Reference triad repeated for every row
pMatA = repmat(refPitches, nPCs * nPCs, 1);

% Single batch call (handles spectral enrichment and deduplication)
fprintf('Computing spectral pitch class similarities for %s reference (N=%d, rho=%.2f)...\n', ...
    refName, nHarm, rho);
simVector = batchCosSimExpTens(pMatA, pMatB, ...
    sigma, r, isRel, isPer, period, ...
    'spectrum', {'harmonic', nHarm, 'powerlaw', rho}, ...
    'verbose', false);

simMatrix = reshape(simVector, nPCs, nPCs);
fprintf('  Done.\n');

%% === Centre the axes on the reference triad ===

rootOffsets  = mod(pcs - refRoot  + 600, 1200) - 600;
thirdOffsets = mod(pcs - refThird + 600, 1200) - 600;

[rootOffsetsSorted, rootSortIdx]   = sort(rootOffsets);
[thirdOffsetsSorted, thirdSortIdx] = sort(thirdOffsets);

simSorted = simMatrix(thirdSortIdx, rootSortIdx);

rootLabels  = noteNames(rootSortIdx);
thirdLabels = noteNames(thirdSortIdx);

%% === Plot ===

figure('Name', sprintf('SPCS: %s reference', refName), ...
       'Position', [100, 100, 700, 600]);

imagesc(rootOffsetsSorted, thirdOffsetsSorted, simSorted);
axis xy;
set(gca, ...
    'XTick', rootOffsetsSorted, 'XTickLabel', rootLabels, ...
    'YTick', thirdOffsetsSorted, 'YTickLabel', thirdLabels, ...
    'TickLength', [0 0]);
xlabel('Root of fifth');
ylabel('Third');
colormap(flipud(gray));  % darker = higher similarity (as in the paper)
cb = colorbar;
cb.Label.String = 'Spectral pitch class similarity';
title(sprintf(['Spectral pitch class similarity: %s reference\n' ...
    '(N = %d harmonics, \\rho = %.2f, \\sigma = %d cents)'], ...
    refName, nHarm, rho, sigma));
daspect([1 1 1]);

%% === Chord labels ===

if showLabels
    hold on;

    for ni = 1:nPCs
        root = pcs(ni);

        % Root offset from reference root
        rx = mod(root - refRoot + 600, 1200) - 600;

        % Major triad: third at root + 400
        majThirdPC = mod(root + 400, 1200);
        majTy = mod(majThirdPC - refThird + 600, 1200) - 600;
        text(rx, majTy, noteNames{ni}, ...
            'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.8 0 0], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

        % Minor triad: third at root + 300
        minThirdPC = mod(root + 300, 1200);
        minTy = mod(minThirdPC - refThird + 600, 1200) - 600;
        minLabel = [lower(noteNames{ni}(1)), noteNames{ni}(2:end)];
        text(rx, minTy, minLabel, ...
            'FontSize', 8, 'FontWeight', 'bold', 'Color', [0 0 0.8], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end

    % Mark the reference triad at the origin
    plot(0, 0, 'ws', 'MarkerSize', 18, 'LineWidth', 2);

    hold off;
end

fprintf('Done.\n');
