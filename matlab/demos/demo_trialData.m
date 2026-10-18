%% demo_trialData.m
%  Adding toolbox predictors to a table of experimental trials.
%
%  Experimental data are usually stored in long format: one row per
%  observation, here one rating by one participant of one stimulus,
%  with the columns that identify the participant, describe the
%  stimulus, and record the response. The analyst wants, on every row,
%  the perceptual predictors of that row's stimulus, so that the table
%  can go straight into a statistical model. This demo takes such a
%  table and adds three predictors to it, as new columns:
%
%    spcs         the spectral pitch-class similarity (SPCS) of the
%                 chord to the scale it was heard in -- a paired
%                 measure, of the stimulus's two parts;
%    harmonicity  the chord's template harmonicity (Milne 2013) -- a
%                 single-set measure, of the chord alone;
%    roughness    the chord's sensory roughness (Sethares 1993).
%
%  It then writes the table out for the statistical analysis.
%
%  The data
%    A simulated experiment: 60 participants rate how well a chord fits a
%    preceding scale, on a 0-100 slider. The stimuli cross three scales
%    with six chord types (five triads and a dominant seventh) at twelve
%    roots, 216 in all. For each scale and chord type, each participant
%    hears a random third of the roots, so 72 stimuli, and hears each of
%    them twice, once in each of two blocks: 144 trials. Each participant
%    also has a Gold-MSI general sophistication score. The table has 60 x
%    144 = 8640 rows. The ratings are generated from a known model (the
%    local function simulateExperiment at the end of this file), so that
%    anyone who fits the model can check that it recovers the parameters.
%    With your own data, the call to simulateExperiment is replaced by T =
%    readtable('myData.csv').
%
%  Repeated stimuli (section 2)
%    Each stimulus occurs on 40 rows on average (twice for each of
%    about 20 participants), so each of its predictor values is needed
%    many times over. The table's columns go straight in nonetheless:
%    every toolbox feature that accepts a 2-D pitch matrix (one row per
%    item, NaN-padded when the items differ in size) finds the repeated
%    rows itself, computes each distinct one once, and returns one
%    value per row.
%
%  Uses: simMaet, templateHarmonicity, and roughness (each batched, with
%        spectral enrichment via its spectrum option),
%        transformAttributes, mptDefaults
%  (from the Music Perception Toolbox).

% The toolbox's one-time informational hints (which route a call took,
% and the like) are switched off for a tidy printout, and restored at
% the end.
prevDefaults = mptDefaults('showHints', false);

%% === User-adjustable parameters ===

% SPCS: sigma = 10 cents, 16 harmonics with power-law rolloff rho = 1;
% r = 1, absolute, periodic at the octave.
prm.sigma  = 10;
prm.r      = 1;
prm.rel    = false;
prm.per    = true;
prm.period = 1200;
prm.spec   = {'harmonic', 16, 'powerlaw', 1};

% Pitch 0 cents is middle C (MIDI 60), which is 6000 in the toolbox's
% absolute cents (100 x MIDI); roughness needs absolute frequencies.
prm.refCents = 6000;

% Where the finished table is written.
outFile = fullfile(tempdir, 'trialData.csv');

%% === 1. The data table ===

T = simulateExperiment(prm, 60, 1);   % with real data: readtable(...)

fprintf('=== 1. The data: %d rows, one per rating ===\n\n', height(T));
disp(head(T, 6));

% The pitch columns, as matrices with one row per trial. Empty cells
% (a triad's fourth note) arrive as NaN, which is the padding the
% toolbox expects.
vars     = T.Properties.VariableNames;
scaleMat = T{:, startsWith(vars, 'scale_')};
chordMat = T{:, startsWith(vars, 'chord_')};

%% === 2. Computing the predictors for every row ===

fprintf('=== 2. Predictors for every row ===\n\n');

% One call per predictor, on all 8640 rows. Each feature computes only
% its distinct rows, and decides what counts as distinct from what its
% value depends on: SPCS by the (scale, chord) pair, harmonicity -- a
% transposition-invariant measure of the chord alone -- by chord type
% regardless of root. So the harmonicity call below computes six
% chords, not 8640.
spcs = simMaet(scaleMat, [], chordMat, [], prm.sigma, prm.r, ...
               prm.rel, prm.per, prm.period, 'spectrum', prm.spec, ...
               'verbose', false);
harm = templateHarmonicity(chordMat, [], prm.sigma, ...
                           'chordSpectrum', prm.spec, 'verbose', false);

% Roughness takes frequencies in Hz, so the chord columns are converted
% first, from absolute cents (NaN padding stays NaN). Roughness
% depends on absolute pitch, so only repeated chords are merged: the
% call below computes 72 chords (6 types at 12 roots).
chordHz = transformAttributes(chordMat + prm.refCents, [], {'cents', 'hz'});
rough   = roughness(chordHz, [], 'spectrum', prm.spec);

% Add the predictors to the table.
T.spcs        = spcs;
T.harmonicity = harm;
T.roughness   = rough;
fprintf('  spcs, harmonicity, and roughness added to all %d rows.\n', ...
        height(T));

%% === 3. Checks before modelling ===

fprintf('\n=== 3. Checks before modelling ===\n\n');

% The predictors' correlations over the stimuli (the first row of each
% stimulus, so that each counts once). Harmonicity and roughness are
% both properties of the chord, and are expected to be correlated.
predNames = {'spcs', 'harmonicity', 'roughness'};
[~, iStim] = unique(T(:, {'scaleName', 'chordName', 'root'}));
C = corrcoef(T{iStim, predNames});
fprintf('  Predictor correlations (over stimuli):\n');
fprintf('    %-12s', ''); fprintf('%12s', predNames{:}); fprintf('\n');
for i = 1:3
    fprintf('    %-12s', predNames{i}); fprintf('%12.2f', C(i, :));
    fprintf('\n');
end

% Per participant: how closely ratings follow SPCS, against Gold-MSI.
% This is descriptive, not the analysis, but it shows the simulated
% interaction: steeper for more sophisticated listeners.
g     = findgroups(T.participant);
gold  = splitapply(@(x) x(1), T.goldMsi, g);
rSpcs = splitapply(@corrOf, T.spcs, T.rating, g);
fprintf(['\n  Per-participant r(rating, SPCS) correlates %.2f with ' ...
         'Gold-MSI.\n'], corrOf(gold, rSpcs));

%% === 4. Writing the table for the analysis ===
%
%  The model itself is fitted in any mixed-effects package. In R, with
%  lme4, and the predictors standardized as in the simulation:
%
%    d <- read.csv("trialData.csv")
%    v <- c("spcs", "harmonicity", "roughness", "goldMsi")
%    d[v] <- scale(d[v])
%    m <- lmer(rating ~ (spcs + harmonicity + roughness) * goldMsi +
%                (1 + spcs + harmonicity + roughness | participant),
%              data = d)
%
%  The fixed effects should come out near the true values given in
%  simulateExperiment. (In MATLAB, fitlme fits the same model, but it
%  needs the Statistics and Machine Learning Toolbox.)

writetable(T, outFile);
fprintf('\n=== 4. Table written to %s ===\n', outFile);

%% === Plot ===

figure('Position', [100 100 1100 420]);

% (a) Mean rating of each stimulus against its SPCS, by chord type.
subplot(1, 2, 1); hold on;
[gs, chordNames] = findgroups(T.chordName);
for k = 1:numel(chordNames)
    in = gs == k;
    gk = findgroups(T.scaleName(in), T.root(in));
    plot(splitapply(@mean, T.spcs(in), gk), ...
         splitapply(@mean, T.rating(in), gk), 'o', 'MarkerSize', 4);
end
hold off; box on;
xlabel('SPCS'); ylabel('mean rating');
title('(a) Each stimulus, averaged over participants');
legend(chordNames, 'Location', 'best', 'FontSize', 8);

% (b) Each participant's rating-SPCS correlation against Gold-MSI.
subplot(1, 2, 2);
plot(gold, rSpcs, 'o');
xlabel('Gold-MSI general sophistication'); ylabel('r(rating, SPCS)');
title('(b) Each participant');

mptDefaults(prevDefaults);
fprintf('\nDone.\n');

%% === Local functions ===

function T = simulateExperiment(prm, nParticipants, seed)
%SIMULATEEXPERIMENT  A stand-in for reading your data.
%
%  This function only manufactures a plausible data table, and can be
%  skipped on a first reading. The ratings are generated from the model
%  that the analysis would fit,
%
%    rating ~ (spcs + harmonicity + roughness) * goldMsi
%             + (1 + spcs + harmonicity + roughness | participant),
%
%  with every predictor standardized (z-scored), and with these true
%  values (rating points per SD of the predictor):
%
%    fixed effects   intercept 50, spcs 8, harmonicity 4,
%                    roughness -3, goldMsi 0, spcs:goldMsi 3,
%                    harmonicity:goldMsi 1, roughness:goldMsi -1
%    participants    SDs of intercept 8, spcs 3, harmonicity 2,
%                    roughness 2 (uncorrelated)
%    residual        SD 10
%
%  So musically sophisticated listeners are simulated as weighting
%  spectral fit more heavily. These values are illustrative, not
%  empirical. Ratings are clipped to the slider's range, which affects
%  about 0.3% of them.

rng(seed);

% --- The stimuli: scale x chord type x root ---
scaleNames = ["diatonic", "harmonic minor", "melodic minor"];
scales = [0 200 400 500 700 900 1100
          0 200 300 500 700 800 1100
          0 200 300 500 700 900 1100];
chordNames = ["major", "minor", "dim", "aug", "sus4", "dom7"];
chords = [0 400 700 NaN; 0 300 700 NaN; 0 300 600 NaN
          0 400 800 NaN; 0 500 700 NaN; 0 400 700 1000];  % NaN-padded
roots = (0:100:1100)';

[ri, ci, si] = ndgrid(1:numel(roots), 1:numel(chordNames), ...
                      1:numel(scaleNames));
ri = ri(:); ci = ci(:); si = si(:);
stim = table(scaleNames(si)', chordNames(ci)', roots(ri), ...
             scales(si, :), chords(ci, :) + roots(ri), ...
             'VariableNames', {'scaleName', 'chordName', 'root', ...
                               'scale', 'chord'});
nStim = height(stim);

% --- The stimuli's predictor values ---
x = [simMaet(stim.scale, [], stim.chord, [], prm.sigma, prm.r, ...
             prm.rel, prm.per, prm.period, 'spectrum', prm.spec, ...
             'verbose', false), ...
     templateHarmonicity(stim.chord, [], prm.sigma, ...
                         'chordSpectrum', prm.spec, 'verbose', false), ...
     roughness(transformAttributes(stim.chord + prm.refCents, [], ...
                                   {'cents', 'hz'}), ...
               [], 'spectrum', prm.spec)];
z = (x - mean(x)) ./ std(x);

% --- The participants ---
% Gold-MSI general sophistication (range 18-126; mean and SD near the
% published norms).
gold  = min(max(round(82 + 21 * randn(nParticipants, 1)), 18), 126);
goldZ = (gold - mean(gold)) / std(gold);
bMain = [8 4 -3];                               % main effects
bGold = [3 1 -1];                               % x goldMsi
u0    = 8 * randn(nParticipants, 1);            % random intercepts
u     = randn(nParticipants, 3) .* [3 2 2];     % random slopes

% --- One row per rating ---
% The stimuli come in cells of 12 roots (one cell per scale and chord
% type, root varying fastest). Each participant hears a random 4 of the
% 12 roots in every cell, and hears that set twice, in a fresh random
% order in each of two blocks.
nCells = nStim / 12;
blocks = cell(2, nParticipants);
for j = 1:nParticipants
    heard = zeros(4, nCells);
    for c = 1:nCells
        heard(:, c) = (c - 1) * 12 + randperm(12, 4)';
    end
    heard  = heard(:);
    slopes = bMain + bGold * goldZ(j) + u(j, :);
    for blk = 1:2
        order  = heard(randperm(numel(heard)));
        n      = numel(order);
        rating = 50 + u0(j) + z(order, :) * slopes' + 10 * randn(n, 1);
        b = stim(order, :);
        b = addvars(b, repmat(j, n, 1), repmat(gold(j), n, 1), ...
                    repmat(blk, n, 1), (blk - 1) * n + (1:n)', ...
                    'Before', 1, 'NewVariableNames', ...
                    {'participant', 'goldMsi', 'block', 'trial'});
        b.rating = min(max(round(rating, 1), 0), 100);
        blocks{blk, j} = b;
    end
end
T = vertcat(blocks{:});           % participant by participant, block 1 first

% As exported by most experiment software: one column per note, empty
% where a chord has fewer notes.
T = splitvars(T, {'scale', 'chord'});
end

function c = corrOf(x, y)
%CORROF  Pearson correlation of two vectors, with base MATLAB only.
R = corrcoef(x, y);
c = R(1, 2);
end
