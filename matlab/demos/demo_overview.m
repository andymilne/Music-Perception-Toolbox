%% demo_overview.m
%  Quick tour of the Music Perception Toolbox.
%
%  The sections run from the simplest material to the richest:
%  expectation tensors of a single multiset, beginning with a chord
%  enriched with spectra and smoothed into a density, multi-attribute
%  expectation tensors (MAETs) of a sequence of events, and then the
%  families of measures that stand beside the tensors -- consonance,
%  balance and evenness, and scale and rhythm structure. Each section ends with a "See also"
%  list of the demos that take its topic further.
%
%  Two running examples recur throughout: the diatonic scale
%  [0, 2, 4, 5, 7, 9, 11] in 12-EDO, and the son clave rhythm
%  [0, 3, 6, 10, 12] in a 16-step cycle. Section 2 combines them, as a
%  diatonic melody set in the clave rhythm.
%
%  Uses: transformAttributes, addSpectra, simMaet, entropyMaet,
%        buildMaet, plotMaet, flatSpecs, packPreMaet, showPreMaet,
%        windowedSimilarity, differenceEvents, templateHarmonicity,
%        tensorHarmonicity, spectralEntropy, roughness, balanceCircular,
%        evennessCircular, dftCircular, coherence, sameness,
%        nTupleEntropy, meanOffset, edges, markovS, circApm, mptDefaults
%  (from the Music Perception Toolbox).
%
%  The Python mirror is demo_overview.py.

clear; clc;

% The toolbox's one-time informational hints (which route a call took,
% and the like) are switched off for a tidy printout, and restored at
% the end.
prevDefaults = mptDefaults('showHints', false);

% The running examples, in the units each section needs.
diat      = [0, 2, 4, 5, 7, 9, 11];                  % 12-EDO steps
diatCents = [0, 200, 400, 500, 700, 900, 1100];
clave     = [0, 3, 6, 10, 12];                       % 16-step cycle

%% === 1. Expectation tensors of a single multiset (User Guide §3.1, §3.3, §6.1, §6.2) ===

% An expectation tensor replaces each element of a multiset with a
% Gaussian of width sigma and sums them, over r-tuples of elements. 1a
% builds one from a chord. Two things are computed from one: the
% similarity of two of them (1b) and the entropy of one (1c). Four
% parameters decide what it represents (1d).

% --- 1a. A chord as a density ---

% A sounded pitch is a spectrum of partials, so spectral enrichment
% replaces each chord tone by its harmonics, weighted here by 1/n
% (powerlaw 1). The enriched chord is then an ordinary multiset: its
% expectation tensor at r = 1, absolute and not periodic, is a density
% over pitch, the chord's smoothed spectrum.
fprintf('\n=== 1a. A C major triad as a spectral density ===\n');
chord = [0; 400; 700];
[p, w] = addSpectra(chord, [], 'harmonic', 8, 'powerlaw', 1);
fprintf('  3 pitches × 8 harmonics = %d partials\n', numel(p));
fprintf('  C''s partials, in cents above C: %s\n', mat2str(round(p(1:8)', 1)));
densChord = buildMaet(p, w, 10, 1, false, false, 0, 'verbose', false);

% Above, each partial's kernel (its weight times a Gaussian of sigma =
% 10 cents); below, the density they sum to. Partials of different tones that land
% close together merge into one peak: C's fifth harmonic (2786 cents
% above C) and E's fourth (2800), for example. Those coincidences are
% what spectral similarity and harmonicity pick up.
figure('Name', 'A chord as a density', 'Position', [100 100 900 550]);
axK = subplot(2, 1, 1);
plotMaet(densChord, 'method', 'kernels', 'axes', axK);
axD = subplot(2, 1, 2);
plotMaet(densChord, 'method', 'density', 'axes', axD);
linkaxes([axK, axD], 'xy');
ylabel(axK, 'kernels');
ylabel(axD, 'density');
xlabel(axD, 'pitch (cents above C)');
set([axK, axD], 'XTick', 0:400:max(p));       % a divisor of the octave
title(axK, 'A C major triad, 8 harmonics per tone, sigma = 10 cents');

% See also:
%   demo_virtualPitches       harmonic templates matched to enriched
%                             chords
%   demo_audioAnalysis        measured spectra (audioPeaks) in place of
%                             synthetic ones
%   demo_jmm_2_3_spectral     spectral enrichment inside a
%                             multi-attribute analysis

% --- 1b. Similarity: spectral pitch class similarity (SPCS) ---

fprintf('\n=== 1b. Spectral pitch class similarity ===\n');
chord_mat = [0, 400, 700;     % Major
             0, 300, 700;     % Minor
             0, 300, 600];    % Dim
chordNames = {'Major', 'Minor', 'Dim'};

% SPCS compares such densities with pitch periodic at the octave
% (isPer = true, period 1200), so each spectrum is folded onto one
% octave of pitch classes before the comparison.
%
% Batched call: the scale (a single row vector) is broadcast against
% every row of chord_mat. The 'spectrum' option does what 1a did by
% hand, enriching both sides identically before the density is built.
s = simMaet(diatCents, [], chord_mat, [], ...
            10, 1, false, true, 1200, ...
            'spectrum', {'harmonic', 24, 'powerlaw', 1}, ...
            'verbose', false);

for i = 1:numel(chordNames)
    fprintf('  Diatonic vs %-5s triad: %.3f\n', chordNames{i}, s(i));
end

% --- 1c. Entropy: how evenly a scale's interval content is spread ---

% A relative dyad tensor (r = 2, relative) is a density over the
% intervals between pairs of pitches, so its entropy is high when a
% scale holds many different intervals in similar numbers and low when
% it holds few. Renyi-2 entropy is computed in closed form, so no grid
% enters.
fprintf('\n=== 1c. Interval-content entropy (Renyi-2, bits) ===\n');
scaleNames = {'Whole-tone', 'Diatonic', 'Chromatic'};
scales     = {0:200:1000, diatCents, 0:100:1100};
for k = 1:numel(scales)
    H = entropyMaet(scales{k}, [], 10, 2, true, true, 1200, ...
                    'method', 'renyi2', 'verbose', false);
    fprintf('  %-10s: %.3f\n', scaleNames{k}, H);
end
% The whole-tone scale holds only even intervals, so its entropy is
% lowest; the chromatic scale holds every interval equally often, so its
% entropy is highest.

% --- 1d. The parameters that define a tensor ---

% The same diatonic scale drawn four ways. The order r sets how many
% elements each point of the density describes, and relative mode
% (isRel) reads a tuple's intervals rather than its pitches, which makes
% the density transposition-invariant and removes one dimension:
% dim = r - isRel. All four are periodic at the octave.
%
%   r = 1, absolute   the pitch classes themselves
%   r = 2, absolute   pairs of pitch classes
%   r = 2, relative   the intervals between pairs: the interval vector
%                     <2, 5, 4, 3, 6, 1>, smoothed and mirrored about
%                     the tritone
%   r = 3, relative   trichords, each drawn as the two intervals above
%                     one of its notes
fprintf('\n=== 1d. Tensor parameters (figure) ===\n');
configs = [1 0; 2 0; 2 1; 3 1];               % [r, isRel] per panel
figure('Name', 'The diatonic scale as four expectation tensors', ...
       'Position', [100 100 900 800]);
for k = 1:size(configs, 1)
    r     = configs(k, 1);
    isRel = logical(configs(k, 2));
    dens  = buildMaet(diatCents, [], 15, r, isRel, true, 1200, ...
                      'verbose', false);
    ax = subplot(2, 2, k);
    plotMaet(dens, 'method', 'density', 'axes', ax);
    dim = r - isRel;
    if isRel
        label = 'interval';   modeStr = 'relative';
    else
        label = 'pitch class'; modeStr = 'absolute';
    end
    title(ax, sprintf('r = %d, %s (dim = %d)', r, modeStr, dim));
    if dim == 1
        xlabel(ax, sprintf('%s (cents)', label));
        ylabel(ax, 'density');
    else
        xlabel(ax, sprintf('%s 1 (cents)', label));
        ylabel(ax, sprintf('%s 2 (cents)', label));
    end
end
fprintf('  Drawn: r = 1 and 2 absolute, r = 2 and 3 relative.\n');

% See also:
%   demo_maetPlots            every combination of r, isRel, isPer, and
%                             isExch, drawn by each of plotMaet's methods
%   demo_triadSpcsGrid        SPCS of every triad containing a fifth
%   demo_edoApprox            how well each n-EDO approximates a JI chord
%   demo_genChainPcs          the same, over generator-chain tunings
%   demo_batchProcessing      a feature for every trial of an experiment
%   demo_dispatchAndKernelControls
%                             speed controls, and Renyi-2 entropy

%% === 2. Multi-attribute expectation tensors (User Guide §3.2, §7.2-7.4) ===

% A MAET takes a sequence of events, each carrying several attributes
% -- here pitch and onset -- and builds one density over all of them
% jointly. The melody below is two cycles of the son clave, each note a
% diatonic pitch; the second cycle is the first transposed up a fifth.
%
%   cycle 1   C  D  E  G  E   at onsets  0  3  6 10 12
%   cycle 2   G  A  B  D  B   at onsets 16 19 22 26 28

% --- 2a. Events and the pre-MAET ---

% A pre-MAET is everything a MAET is built from: the values of each
% attribute for each event, their weights, and each attribute's
% parameters. Pitch is periodic at the octave, with sigma = 20 cents;
% onset is not periodic, with sigma = 0.5 steps. Pitch is given in MIDI
% note numbers and converted to cents, the units its sigma is read in:
% transformAttributes rescales an attribute (MIDI, Hz, cents, ERB-rate,
% or a logarithm), and a logarithmic rescaling turns a proportional
% change -- a tempo change, say -- into a common shift.
fprintf('\n=== 2a. A melody as a pre-MAET ===\n');
onsets = [clave, clave + 16];
midi   = [60 62 64 67 64 67 69 71 74 71];
pitch  = transformAttributes(midi, [], {'midi', 'cents'});
% See also:
%   demo_preprocessing        transformAttributes (attribute
%                             rescaling) among the other pre-MAET
%                             preprocessing operations, and how a
%                             rescaling carries sigma with it
%   demo_repetitionHandling   a non-linear rescaling (log step size)

pAttr = {pitch, onsets};                        % one row per attribute
specs = flatSpecs(pAttr, 'name', {'pitch', 'onset'}, ...
                  'sigma', [20 0.5], 'isPer', [true false], ...
                  'period', [1200 0]);
melody = packPreMaet(pAttr, [], specs);
showPreMaet(melody, 'maxEvents', []);

% The query: the melody's opening three notes, C D E.
query = packPreMaet({pitch(1:3), onsets(1:3)}, [], specs);

% --- 2b. Where does the motif occur? ---

% windowedSimilarity translates the query along the onset attribute by each
% offset, windows the melody around it, and compares the two in pitch
% and onset jointly. The query and the melody are both written from
% onset 0, so an offset is the onset at which the query starts in the
% melody: 0 is the query where it was taken from.
fprintf('\n=== 2b. Motif search, absolute pitch ===\n');
offsets = -4:0.5:24;
window  = {'gaussian', 16};     % one clave cycle, equivalent width
sAbs = windowedSimilarity(melody, query, [], 'offsets', offsets, ...
                          'windowAttr', 2, 'contextWindow', window);
reportPeaks(offsets, sAbs, 0.5);
% One peak, at offset 0 -- the query's own position. The
% transposed statement in cycle 2 is not found: in absolute mode G A B
% is not C D E.

% --- 2c. Invariance by preprocessing ---

% Differencing the pitch attribute replaces each pitch with the
% interval from the previous one, so a transposed statement has the
% same values as the original. Onset is passed through (order 0),
% keeping each interval at the onset of its second note. Differencing
% drops the first event but leaves every surviving onset where it was,
% so an offset still says where the original query starts, and the two
% profiles share one horizontal axis.
% The pitch sigma grows by sqrt(2), since a difference of two uncertain
% values is less certain than either; differenceEvents announces this.
fprintf('\n=== 2c. Motif search, pitch intervals ===\n');
melodyD = differenceEvents(melody, [1 0]);
queryD  = differenceEvents(query, [1 0]);
showPreMaet(melodyD, 'maxEvents', []);
sDiff = windowedSimilarity(melodyD, queryD, [], 'offsets', offsets, ...
                           'windowAttr', 2, 'contextWindow', window);
reportPeaks(offsets, sDiff, 0.5);
% Two peaks of equal height, at offsets 0 and 16 -- one per clave
% cycle: the rising pair of whole
% tones is found in both. Which preprocessing and which mode are chosen
% is what decides what counts as "the same".

% Top: the melody as a piano roll, the query's notes filled. Bottom: the
% two profiles against the onset of the query's first note, on the same
% horizontal axis, so each peak sits under the statement it found.
figure('Name', 'Motif search', 'Position', [100 100 900 600]);
cols = lines(2);
noteNames = {'C', 'D', 'E', 'F', 'G', 'A', 'B'};
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
scatter(ax1, onsets, midi, 60, cols(1, :), 'LineWidth', 1.5, ...
        'DisplayName', 'melody');
scatter(ax1, onsets(1:3), midi(1:3), 60, cols(1, :), 'filled', ...
        'DisplayName', 'query (C D E)');
for k = 1:numel(onsets)
    text(ax1, onsets(k), midi(k) + 1.2, ...
         noteNames{[0 2 4 5 7 9 11] == mod(midi(k), 12)}, ...
         'HorizontalAlignment', 'center', 'FontSize', 8);
end
xline(ax1, 16, '--', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
hold(ax1, 'off');
ylabel(ax1, 'MIDI pitch');
ylim(ax1, [58 77]);
title(ax1, 'Two clave cycles, the second a fifth higher');
legend(ax1, 'Location', 'northwest');
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');
plot(ax2, offsets, sAbs, 'LineWidth', 2, 'DisplayName', 'absolute pitch');
plot(ax2, offsets, sDiff, 'LineWidth', 2, ...
     'DisplayName', 'pitch intervals (differenced)');
xline(ax2, 16, '--', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
hold(ax2, 'off');
xlabel(ax2, 'onset of the query''s first note (steps)');
ylabel(ax2, 'similarity to C D E');
title(ax2, 'Where does the opening motif recur?');
legend(ax2, 'Location', 'northeast');
linkaxes([ax1 ax2], 'x');
xlim(ax2, [-5 29]);

% See also:
%   demo_preMaetIo            showing, exporting, and importing a
%                             pre-MAET
%   demo_preprocessing        the pre-MAET preprocessing operations,
%                             and their compositions
%   demo_scoreWorkflow        a pre-MAET read from MusicXML or MIDI
%                             (then demo_scoreGrid,
%                             demo_scoreCategoricals)
%   demo_translateSweep       a query swept in pitch and time at once
%   demo_tempoInvariance      motif search tolerant of tempo change
%   demo_repetitionHandling   interval-scale invariance, and what to do
%                             with repeated notes
%   demo_helixBlend           pitch read as pitch class and height at
%                             once
%   jmm/                      the analyses of the JMM article: entropy
%                             across a chorale (1.1), voice-aware
%                             similarity (1.2), cadence finding (1.3),
%                             tuple size in chord matching (1.4), motif
%                             discovery (2.1, 2.2), a motif found in any
%                             key under spectral enrichment (2.3),
%                             phase in Piano Phase (3.1-3.3), and an
%                             expert analysis carried as a nested
%                             attribute (4.1); see jmm/README.md

%% === 3. Consonance and harmonicity (User Guide §6.3) ===

fprintf('\n=== 3. Harmonicity and entropy (JI major triad) ===\n');
ji_triad = [0, 386.31, 701.96];
spec = {'harmonic', 24, 'powerlaw', 1};

[hMax, hEnt] = templateHarmonicity(ji_triad, [], 12, ...
    'chordSpectrum', spec);
fprintf('  Template harmonicity (hMax):     %.4f\n', hMax);
fprintf('  Template harmonicity (hEntropy): %.4f\n', hEnt);

h = tensorHarmonicity(ji_triad, [], 12, 'spectrum', spec);
fprintf('  Tensor harmonicity:              %.4f\n', h);

H = spectralEntropy(ji_triad, [], 12, 'spectrum', spec);
fprintf('  Spectral entropy:                %.4f\n', H);

fprintf('\n=== JI vs 12-EDO comparison ===\n');
% Both features take a 2-D matrix, one chord per row, and return one
% value per row (and deduplicate repeated rows internally).
edo_triad = [0, 400, 700];
triads = [ji_triad; edo_triad];
[hMaxBoth, ~] = templateHarmonicity(triads, [], 12, 'chordSpectrum', spec);
HBoth = spectralEntropy(triads, [], 12, 'spectrum', spec);
names = {'JI', '12-EDO'};
for k = 1:2
    fprintf('  %-6s  hMax=%.4f  specEntropy=%.4f\n', names{k}, hMaxBoth(k), HBoth(k));
end

fprintf('\n=== Roughness ===\n');
p_cents = transformAttributes([60, 64, 67], [], {'midi', 'cents'});
[p_r, w_r] = addSpectra(p_cents, [], 'harmonic', 8, 'powerlaw', 1);
f_hz = transformAttributes(p_r, [], {'cents', 'hz'});
r = roughness(f_hz, w_r);
fprintf('  C major triad (8 harmonics): roughness = %.4f\n', r);

% See also:
%   demo_triadConsonance      every measure over a grid of triads
%   demo_virtualPitches       the salience profiles behind template
%                             harmonicity
%   demo_audioAnalysis        the same measures from recorded sounds
%   demo_batchProcessing      the same measures for a table of trials

%% === 4. Balance and evenness (User Guide §6.4) ===

fprintf('\n=== 4. Balance and evenness ===\n');

fprintf('  Diatonic scale [0,2,4,5,7,9,11] in 12-EDO:\n');
fprintf('    Balance:  %.3f\n', balanceCircular(diat, [], 12));
fprintf('    Evenness: %.3f\n', evennessCircular(diat, 12));

fprintf('  Son clave [0,3,6,10,12] in 16:\n');
fprintf('    Balance:  %.3f\n', balanceCircular(clave, [], 16));
fprintf('    Evenness: %.3f\n', evennessCircular(clave, 16));

% Balance places each element on the unit circle, at angle 2*pi*p/period,
% and is 1 minus the length of their mean: a collection whose elements
% pull equally in every direction balances at the centre. The arrow is
% that mean, the first coefficient dftCircular returns. It is barely visible for the diatonic scale and the clave,
% which are both close to perfectly balanced; the third panel, five
% onsets packed into the first half of the cycle, shows what an
% unbalanced rhythm looks like.
figure('Name', 'Balance', 'Position', [100 100 1200 450]);
panelNames   = {'Diatonic scale', 'Son clave', 'First half only'};
panelPts     = {diat, clave, [0 2 4 6 8]};
panelPeriods = [12 16 16];
cols = lines(2);
for k = 1:3
    ax = subplot(1, 3, k);
    hold(ax, 'on');
    ring = linspace(0, 2 * pi, 361);
    plot(ax, sin(ring), cos(ring), 'Color', [0.8 0.8 0.8]);
    P = panelPeriods(k);
    tk = 2 * pi * (0:P-1) / P;
    scatter(ax, sin(tk), cos(tk), 10, [0.7 0.7 0.7], 'filled');
    pts = panelPts{k};
    ang = 2 * pi * pts / P;
    scatter(ax, sin(ang), cos(ang), 70, cols(1, :), 'filled');
    for j = 1:numel(pts)
        text(ax, 1.18 * sin(ang(j)), 1.18 * cos(ang(j)), ...
             num2str(pts(j)), 'HorizontalAlignment', 'center');
    end
    F = dftCircular(pts, [], P);
    m = F(1);                           % the mean, as cos + i sin
    quiver(ax, 0, 0, imag(m), real(m), 0, 'Color', [0.84 0.15 0.16], ...
           'LineWidth', 2, 'MaxHeadSize', 0.5);
    plot(ax, 0, 0, 'k+');
    hold(ax, 'off');
    title(ax, sprintf('%s\nbalance = %.3f', panelNames{k}, ...
                      balanceCircular(pts, [], P)));
    axis(ax, 'equal', 'off');
    xlim(ax, [-1.35 1.35]);
    ylim(ax, [-1.35 1.35]);
end
sgtitle('Balance: the mean of the elements on the circle (red)');

% See also:
%   demo_dftCircularSimulate  balance and evenness under positional
%                             uncertainty (sigma > 0), analytically and
%                             by Monte Carlo

%% === 5. Scale and rhythm structure (User Guide §6.5) ===

fprintf('\n=== 5. Scale structure (diatonic) ===\n');
[c, nc] = coherence(diat, 12);
[sq, nd] = sameness(diat, 12);
fprintf('  Coherence: %.3f (%d failure)\n', c, nc);
fprintf('  Sameness:  %.3f (%d ambiguity)\n', sq, nd);

H1 = nTupleEntropy(diat, 12, 1);
H2 = nTupleEntropy(diat, 12, 2);
fprintf('  1-tuple entropy: %.3f\n', H1);
fprintf('  2-tuple entropy: %.3f\n', H2);

% Mode brightness. meanOffset, read at a query point, sums the arcs
% from that point up to each pitch class of the scale and subtracts the
% arcs down to them, so at a mode's tonic it says how high the mode's
% pitches sit above its tonic. The seven modes of the diatonic scale,
% their tonics taken down the chain of fifths from F:
modes  = {'Lydian', 'Ionian', 'Mixolydian', 'Dorian', 'Aeolian', ...
          'Phrygian', 'Locrian'};
tonics = [5 0 7 2 9 4 11];
bright = meanOffset(diat, [], 12, tonics);
fprintf('  Mode brightness (mean offset at the tonic):\n');
for k = 1:numel(modes)
    fprintf('    %-10s %+.3f\n', modes{k}, bright(k));
end
% Each step down the chain lowers one degree of the mode by a semitone,
% so brightness falls by 1/6 at each step, from Lydian to Locrian.

fprintf('\n=== Rhythm structure (son clave) ===\n');
[c, nc] = coherence(clave, 16);
[sq, nd] = sameness(clave, 16);
fprintf('  Coherence: %.3f (%d failures)\n', c, nc);
fprintf('  Sameness:  %.3f (%d ambiguities)\n', sq, nd);

H1 = nTupleEntropy(clave, 16, 1);
H2 = nTupleEntropy(clave, 16, 2);
fprintf('  1-tuple entropy: %.3f\n', H1);
fprintf('  2-tuple entropy: %.3f\n', H2);

h = meanOffset(clave, [], 16);
fprintf('  Mean offset: %s\n', mat2str(round(h, 3)));

edg = edges(clave, [], 16);
fprintf('  Edges: %s\n', mat2str(round(edg, 3)));

y = markovS(clave, [], 16);
fprintf('  Markov(3): %s\n', mat2str(round(y, 3)));

% The circular autocorrelation phase matrix (APM) holds, for each lag
% and phase, how many pairs of onsets fall on successive beats of the
% pulse with that lag, started at that phase. Summed over lags, it
% gives each position a metrical weight: the phase sum.
[~, apmPhase] = circApm(clave, [], 16);
fprintf('  APM phase sum: %s\n', mat2str(round(apmPhase, 3)));
% Onsets 0, 6, 10, and 12 carry the most (48); onset 3, off every
% strong pulse, carries half as much (24).

% The four position-wise measures, one value per pulse of the clave's
% cycle. Pulses holding an onset are drawn dark.
figure('Name', 'Rhythm structure', 'Position', [100 100 900 900]);
pulses  = 0:15;
isOnset = ismember(pulses, clave);
darkCol  = [0.12 0.47 0.71];
lightCol = [0.65 0.78 0.90];
barCols = repmat(lightCol, 16, 1);
barCols(isOnset, :) = repmat(darkCol, nnz(isOnset), 1);
vals = {h, edg, y, apmPhase};
labs = {'mean offset', 'edges', 'Markov(3)', 'APM phase sum'};
for k = 1:4
    ax = subplot(4, 1, k);
    b = bar(ax, pulses, vals{k}, 0.7, 'FaceColor', 'flat', ...
            'EdgeColor', 'none');
    b.CData = barCols;
    yline(ax, 0, 'Color', [0.5 0.5 0.5]);
    ylabel(ax, labs{k});
    xticks(ax, pulses);
    if k == 1
        title(ax, 'Son clave: position-wise measures (onsets dark)');
    end
end
xlabel(ax, 'pulse');

% See also:
%   demo_sigmaSpace           soft (sigma > 0) coherence, sameness, and
%                             n-tuple entropy, and what sigma stands for

mptDefaults(prevDefaults);
fprintf('\nDone.\n');

%% === Local functions ===

function reportPeaks(offsets, S, thresh)
%REPORTPEAKS Print the local maxima of a similarity profile above thresh.
    S = S(:).';
    for i = 2:numel(S) - 1
        if S(i) > thresh && S(i) >= S(i - 1) && S(i) > S(i + 1)
            fprintf('  peak at offset %4.1f steps: %.3f\n', offsets(i), S(i));
        end
    end
end
