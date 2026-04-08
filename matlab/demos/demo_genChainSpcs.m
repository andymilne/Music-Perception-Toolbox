%% demo_genChainSpcs.m
%  Spectral pitch class similarity (SPCS) of generator-chain tunings
%  against a reference chord, as the generator interval is swept
%  continuously.
%
%  A generator-chain of cardinality n is a scale formed by stacking n
%  copies of a generating interval, reduced modulo the period:
%    pitches = mod((0:n-1) * gen, period)
%
%  For each generator size (swept in fine steps from 0 to period/2), the
%  SPCS between the resulting scale and a reference chord is computed.
%  Results are reflected to fill the full range [0, period), since a
%  generator of size gen produces the same scale as period - gen.
%  The plot reveals which generator sizes produce scales that are most
%  similar to the reference chord — peaks correspond to tunings whose
%  intervals closely approximate those of the chord.
%
%  Examples of this type of plot appear as Examples 6.4 and 6.5 / Figures
%  5-7 in:
%    Milne, A. J., Sethares, W. A., Laney, R., & Sharp, D. B. (2011).
%    Modelling the similarity of pitch collections with expectation tensors.
%    Journal of Mathematics and Music, 5(1), 1-20.
%  (That paper uses the term "beta-chain" for what is here called a
%  generator-chain, with "beta" denoting the generator interval.)
%
%  Uses: batchCosSimExpTens, cosSimExpTens
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% Reference chord (in cents)
%   4:5:6 JI major triad: [0, 386.31, 701.96] with period 1200
%   Bohlen-Pierce "major" triad: [0, 884.36, 1466.87] with period 1902
refPitches = [0, 1200*log2(5/4), 1200*log2(6/4)];
refWeights = [];   % weights for reference pitches (empty = uniform)
refName    = '4:5:6 JI major triad';

% Generator-chain parameters
nTones  = 19;      % number of tones in the chain
genStep = 0.1;     % step size for generator sweep (cents)

% Expectation tensor parameters
sigma  = 10;      % Gaussian smoothing width (cents)
r      = 2;       % dyad expectation tensor
isRel  = 1;       % relative (transposition-invariant)
isPer  = 1;       % periodic (pitch-class equivalence)
period = 1200;    % period in cents (1200 = octave; 1902 = tritave)

% Display options
invertCircular = true;   % if true, higher similarity is towards the centre
                         % (as in the article, which plots distance);
                         % if false, higher similarity is at the perimeter
nLabels        = 15;     % maximum number of peak labels on each plot
minLabelSpacing = 9;    % minimum spacing between labels (cents); increase
                         % to reduce clutter, decrease to show more peaks

%% === Build pitch matrices (0 to period/2 only) ===
% A generator-chain with generator gen produces the same scale as one with
% generator period - gen (stacking upward vs downward), so we compute only
% the first half and reflect the results.

halfRange = 0:genStep:(period / 2);
nHalf     = numel(halfRange);

% Reference chord: same for every row
pitchesA = repmat(refPitches, nHalf, 1);

% Reference weights
if ~isempty(refWeights)
    weightsA = repmat(refWeights, nHalf, 1);
else
    weightsA = [];
end

% Generator-chain pitch sets: each row is a different generator value
pitchesB = NaN(nHalf, nTones);
for i = 1:nHalf
    gen = halfRange(i);
    pitchesB(i, :) = mod((0:nTones-1) * gen, period);
end

%% === Compute similarities ===

fprintf('Computing SPCS of %d-tone generator-chains (gen = 0 to %.1f, step %.2f) against %s...\n', ...
    nTones, period / 2, genStep, refName);
opts = {'verbose', false};
if ~isempty(weightsA)
    opts = [{'weightsA', weightsA}, opts];
end
sHalf = batchCosSimExpTens(pitchesA, pitchesB, ...
    sigma, r, isRel, isPer, period, opts{:});
fprintf('Done.\n');

% Round to 3 decimal places for display (avoids floating-point artifacts
% such as values marginally above 1 or below 0)
sHalf = round(sHalf, 3);

% Reflect to produce the full range [0, period)
% gen and (period - gen) give identical scales, so the second half is
% simply sHalf reversed, excluding both endpoints to avoid duplication.
genRange = 0:genStep:(period - genStep);
s = [sHalf(:); flipud(sHalf(2:end-1))];

% If genStep does not divide period/2 evenly, the reflected vector may be
% slightly shorter or longer than genRange. Pad or trim to match.
if numel(s) ~= numel(genRange)
    warning(['genStep (%.4g) does not divide period/2 (%.4g) evenly — ' ...
        'steps will be slightly uneven near the midpoint.'], genStep, period / 2);
    if numel(s) < numel(genRange)
        s(end+1:numel(genRange)) = s(end);
    else
        s = s(1:numel(genRange));
    end
end

%% === Find peaks ===

[pks, locs] = findpeaks(s, 'MinPeakProminence', 0.01, 'SortStr', 'descend');

%% === Linear plot ===

figure('Name', 'Generator-chain SPCS', ...
       'Position', [100, 100, 1000, 400]);

plot(genRange, s, 'Color', [0.2 0.2 0.6], 'LineWidth', 0.8);
xlabel('Generator (cents)');
ylabel('Spectral pitch class similarity');
title(sprintf(['SPCS of %d-tone generator-chains with %s\n' ...
    '(r = %d, isRel = %d, \\sigma = %d cents, period = %d cents)'], ...
    nTones, refName, r, isRel, sigma, period));
xlim([0, period]);

hold on;
labelledGens = [];
labelled = 0;
for li = 1:numel(pks)
    genVal = genRange(locs(li));

    % Skip if too close to an already-labelled peak
    if ~isempty(labelledGens) && any(abs(labelledGens - genVal) < minLabelSpacing)
        continue;
    end

    plot(genVal, pks(li), 'r.', 'MarkerSize', 3);
    text(genVal, pks(li), sprintf('  %.1f', genVal), ...
        'FontSize', 7, 'Color', [0.8 0 0], ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom');

    labelledGens = [labelledGens, genVal]; %#ok<AGROW>
    labelled = labelled + 1;
    if labelled >= nLabels
        break;
    end
end
hold off;

%% === Circular plot ===

% Map generator to angle: 0 at the top (12 o'clock), increasing clockwise
theta = 2 * pi * genRange / period;

% Close the curve by appending the first point
thetaC = [theta(:); theta(1)];
sC     = [s(:); s(1)];

figure('Name', 'Generator-chain SPCS (circular)', ...
       'Position', [100, 550, 600, 600]);

% SPCS as radius, scaled to fixed [0, 1] range:
%   invertCircular: SPCS = 0 at circumference (r = 1), SPCS = 1 at centre (r = 0)
%   normal:         SPCS = 0 at centre (r = 0), SPCS = 1 at circumference (r = 1)
if invertCircular
    rC = 1 - sC;
else
    rC = sC;
end

polarplot(thetaC, rC, 'Color', [0.2 0.2 0.6], 'LineWidth', 1);
ax = gca;
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'clockwise';
ax.RLim = [0, 1];

% Tick labels in cents rather than degrees
nTicks = 12;
tickAngles = linspace(0, 360 - 360/nTicks, nTicks);
tickLabels = arrayfun(@(a) sprintf('%.0f', a * period / 360), ...
    tickAngles, 'UniformOutput', false);
ax.ThetaTick = tickAngles;
ax.ThetaTickLabel = tickLabels;

% Hide radial tick labels (the absolute radius values are not meaningful)
ax.RTickLabel = [];

if invertCircular
    titleExtra = 'centre = most similar';
else
    titleExtra = 'perimeter = most similar';
end
title(sprintf(['SPCS of %d-tone generator-chains with %s\n' ...
    '(r = %d, isRel = %d, \\sigma = %d cents, period = %d cents; %s)'], ...
    nTones, refName, r, isRel, sigma, period, titleExtra));

% Label top peaks on the circular plot
hold on;
labelledGensC = [];
labelledC = 0;
for li = 1:numel(pks)
    genVal = genRange(locs(li));
    if ~isempty(labelledGensC) && any(abs(labelledGensC - genVal) < minLabelSpacing)
        continue;
    end

    peakTheta = 2 * pi * genVal / period;
    if invertCircular
        peakR  = 1 - pks(li);
        labelR = max(0, peakR - 0.05);
    else
        peakR  = pks(li);
        labelR = min(1, peakR + 0.05);
    end
    polarplot(peakTheta, peakR, 'r.', 'MarkerSize', 3);

    text(peakTheta, labelR, sprintf('%.1f', genVal), ...
        'FontSize', 7, 'Color', [0.8 0 0], ...
        'HorizontalAlignment', 'center');

    labelledGensC = [labelledGensC, genVal]; %#ok<AGROW>
    labelledC = labelledC + 1;
    if labelledC >= nLabels
        break;
    end
end
hold off;

%% === Console output: top peaks ===
% Show only generators in [0, period/2] to avoid duplication (gen and
% period - gen produce the same scale and hence the same SPCS; any tiny
% difference is due to floating-point arithmetic). The complement
% (period - gen) is shown in parentheses.

% For each peak, take the best SPCS of gen and period-gen
halfPeriod = period / 2;
fprintf('\nTop peaks (generator values with highest SPCS):\n');
fprintf('%-14s  %-14s  %s\n', 'gen (cents)', '(complement)', 'SPCS');
fprintf('%s\n', repmat('-', 1, 42));
labelledGensT = [];
printed = 0;
for li = 1:numel(pks)
    genVal = genRange(locs(li));

    % Fold to [0, period/2]
    if genVal > halfPeriod
        genVal = period - genVal;
    end
    complement = period - genVal;

    % Skip if too close to an already-printed generator
    if ~isempty(labelledGensT) && any(abs(labelledGensT - genVal) < minLabelSpacing)
        continue;
    end

    % Best SPCS of gen and its complement (should be identical up to
    % floating-point error)
    idxGen  = round(genVal / genStep) + 1;
    idxComp = round(complement / genStep) + 1;
    idxGen  = max(1, min(idxGen, numel(s)));
    idxComp = max(1, min(idxComp, numel(s)));
    bestSpcs = max(s(idxGen), s(idxComp));

    fprintf('%-14.1f  (%-10.1f)  %.3f\n', genVal, complement, bestSpcs);
    labelledGensT = [labelledGensT, genVal]; %#ok<AGROW>
    printed = printed + 1;
    if printed >= nLabels
        break;
    end
end