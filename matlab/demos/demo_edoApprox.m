%% demo_edoApprox.m
%  Spectral pitch class similarity (SPCS) of equal divisions of the octave
%  (n-EDOs) to a just intonation reference chord, using relative dyad
%  expectation tensors (r = 2, isRel = 1, dim = 1).
%
%  An example of this type of plot appears as Example 6.3 / Figure 4 in:
%    Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
%    Modelling the similarity of pitch collections with expectation tensors.
%    Journal of Mathematics and Music, 5(1), 1-20.
%
%  For each value of n, the n-EDO pitch set {0, 1200/n, 2*1200/n, ...,
%  (n-1)*1200/n} is compared to the reference chord. EDOs whose pitch
%  classes include good approximations to the chord's intervals will have
%  higher similarity.
%
%  Because r = 2 and isRel = 1, the effective dimensionality is 1 (i.e.,
%  the expectation tensor is a one-dimensional density over intervals).
%  This is what makes these "one-dimensional approximations."
%
%  Uses: batchCosSimExpTens, cosSimExpTens
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% Reference chord in just intonation (cents)
%   4:5:6 major triad: [0, 386.31, 701.96]
%   5:6:7 subminor triad: [0, 315.64, 582.51]
%   4:5:6:7 dominant seventh: [0, 386.31, 701.96, 968.83]
refPitches = [0, 1200*log2(5/4), 1200*log2(6/4)];
refWeights = [];   % weights for reference pitches (empty = uniform;
                   % if specified, must be same length as refPitches)
refName    = '4:5:6 JI major triad';

% Range of EDOs to test
nMin = 2;
nMax = 102;

% Expectation tensor parameters
sigma  = 10;      % Gaussian smoothing width (cents)
r      = 2;       % dyad expectation tensor
isRel  = 1;       % relative (transposition-invariant)
isPer  = 1;       % periodic (pitch-class equivalence)
period = 1200;    % one octave in cents

%% === Build pitch matrices ===

edoRange = nMin:nMax;
nEDOs    = numel(edoRange);
maxN     = nMax;  % maximum number of pitches in any EDO

% Reference chord: same for every row
pitchesA = repmat(refPitches, nEDOs, 1);

% Reference weights: same for every row (empty = uniform)
if ~isempty(refWeights)
    weightsA = repmat(refWeights, nEDOs, 1);
else
    weightsA = [];
end

% EDO pitch sets: NaN-padded to maxN columns
pitchesB = NaN(nEDOs, maxN);
for i = 1:nEDOs
    n = edoRange(i);
    edoPitches = (0:n-1) * (1200 / n);
    pitchesB(i, 1:n) = edoPitches;
end

%% === Compute similarities ===

fprintf('Computing SPCS of %d EDOs against %s...\n', nEDOs, refName);
opts = {'verbose', false};
if ~isempty(weightsA)
    opts = [{'weightsA', weightsA}, opts];
end
s = batchCosSimExpTens(pitchesA, pitchesB, ...
    sigma, r, isRel, isPer, period, opts{:});
fprintf('Done.\n');

% Round to 3 decimal places for display
s = round(s, 3);

%% === Plot ===

figure('Name', 'EDO approximation quality', ...
       'Position', [100, 100, 900, 400]);

% Stem plot: emphasises the discrete nature of EDOs
stem(edoRange, s, 'filled', 'MarkerSize', 4, 'LineWidth', 0.8, ...
    'Color', [0.2 0.2 0.6]);
xlabel('n-EDO');
ylabel('Spectral pitch class similarity');
title(sprintf(['SPCS of n-EDOs with %s\n' ...
    '(r = %d, isRel = %d, \\sigma = %d cents)'], ...
    refName, r, isRel, sigma));
xlim([nMin - 1, nMax + 1]);

% Label the top peaks
[~, sortIdx] = sort(s, 'descend');
nLabels = 10;
hold on;
for li = 1:min(nLabels, nEDOs)
    idx = sortIdx(li);
    n   = edoRange(idx);
    text(n, s(idx), sprintf('  %d', n), ...
        'FontSize', 8, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom');
end
hold off;

%% === Console output: top EDOs ===

fprintf('\nTop %d EDOs by SPCS with %s:\n', nLabels, refName);
fprintf('%-8s  %s\n', 'n-EDO', 'SPCS');
fprintf('%s\n', repmat('-', 1, 20));
for li = 1:min(nLabels, nEDOs)
    idx = sortIdx(li);
    fprintf('%-8d  %.3f\n', edoRange(idx), s(idx));
end