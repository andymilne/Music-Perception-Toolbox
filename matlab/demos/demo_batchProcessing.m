%% demo_batchProcessing.m
%  Analysing experimental data: perceptual features for a table of trials.
%
%  A typical experiment presents a stimulus per trial and the analyst
%  wants one or more perceptual predictors for every trial, aligned with
%  the responses. This demo builds a synthetic trial table — 3 scales x 4
%  chord types x 12 root transpositions = 144 trials — and computes, for
%  every trial, a paired measure (the spectral pitch-class similarity of
%  the chord to its scale, SPCS) and several single-set measures of the
%  chord (spectral entropy, template harmonicity, tensor harmonicity, and
%  roughness), then tabulates and plots them.
%
%  The point of method is that the trial table goes straight in. Every
%  toolbox feature that accepts a 2-D pitch matrix (one row per trial,
%  NaN-padded when the chords differ in size) — simMaet in its
%  batched-raw mode, spectralEntropy, templateHarmonicity,
%  tensorHarmonicity, virtualPitches — deduplicates its rows internally
%  by a canonical key, so the 144 chord rows here cost 4 chord-type
%  computations, and the 144 (scale, chord) pairs only as many distinct
%  pairs as there are. No manual unique() step is needed.
%
%  The deduplication follows the analysis: each feature keys its rows by
%  what its value depends on. The single-set features (spectral
%  entropy, and template and tensor harmonicity) are
%  transposition-invariant measures of a chord and take no mode flags,
%  so their key ignores transposition and pitch order: the twelve
%  transpositions of a chord type collapse to one computation, and the
%  144 chord rows to 4. SPCS depends on where the chord lies against its
%  scale, so its key is the (scale, chord) pair, up to transposing both
%  together, under the analysis parameters (sigma, r, isRel, isPer,
%  period). Here, with isPer = 1, pitch is read as pitch class, so the
%  augmented triad's transpositions by a major third, which give one
%  pitch-class set, collapse, and the 144 pairs cost 120 computations;
%  under isRel = 1 (which needs r >= 2) every transposition of a chord
%  would collapse, a relative density being transposition-invariant.
%  The analyst changes the flags and the saving follows, with no change
%  to the calling code. The one feature without a batched form,
%  roughness (which depends on absolute frequency and so cannot share
%  work across transpositions), is looped over the distinct rows.
%
%  Workflow 3 shows a second, quite different sense of "batch". Rows of a
%  2-D pitch matrix are single multisets over one attribute, and what
%  Workflows 1 and 2 exploit is deduplication *within* such a matrix. A
%  multi-attribute analysis has no rows to deduplicate: each item is
%  a whole pre-MAET. Batching there means passing a *cell* of them where
%  a cell of densities would go, which loops rather than collapses — the
%  saving is in the calling code, not in the arithmetic.
%
%  Uses: simMaet, spectralEntropy, templateHarmonicity,
%        tensorHarmonicity, addSpectra, roughness, transformAttributes,
%        packPreMaet, flatSpecs, translateAttributes,
%        sweepSimMaet, buildMaet, mptDefaults
%  (from the Music Perception Toolbox).
%
%  The Python mirror is demo_batch_processing.py.

% The toolbox's one-time informational hints (which route a call took,
% and the like) are switched off for a tidy printout, and restored at
% the end.
prevDefaults = mptDefaults('showHints', false);

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
%  WORKFLOW 1: Paired measure (SPCS) via batched simMaet
%  Two 2-D matrices, one row per trial, dispatch to batched-raw mode;
%  repeated rows and repeated (scale, chord) pairs are deduplicated
%  internally, and the spectrum is applied inside the call.
%  =====================================================================

fprintf('=== Workflow 1: SPCS via batched simMaet ===\n\n');

spcs = simMaet(pMatA, [], pMatB, [], ...
    sigma, r, isRel, isPer, period, ...
    'spectrum', spec);

spcs = round(spcs, 3);

% The rows were laid out scale by chord type by root (root fastest), so
% the profile reshapes straight into a scale x chord x root array.
spcsGrid = permute(reshape(spcs, nRoots, nChords, nScales), [3 2 1]);

% Display as scale × chord × root tables
for si = 1:nScales
    fprintf('\n  %s:\n', scaleNames{si});
    fprintf('  %-8s', '');
    fprintf('%6d', roots);
    fprintf('\n');

    for ci = 1:nChords
        fprintf('  %-8s', chordTypeNames{ci});
        fprintf('%6.3f', squeeze(spcsGrid(si, ci, :)));
        fprintf('\n');
    end
end

%% =====================================================================
%  WORKFLOW 2: Single-set measures on the trial table
%
%  The batched features take the 144-row chord matrix as it is: each
%  deduplicates its rows internally (a canonical key invariant to
%  transposition and pitch order, so the 12 roots x 4 types collapse to
%  4 computations) and returns one value per trial. Each applies the
%  spectrum through its own argument; pre-enriching all pitches would
%  be prohibitively expensive for tensor harmonicity with many partials.
%  =====================================================================

fprintf('\n=== Workflow 2: Single-set measures (chord features) ===\n\n');

specEnt        = spectralEntropy(pMatB, [], sigma, 'spectrum', spec);
[hMax, hEnt]   = templateHarmonicity(pMatB, [], sigma, 'chordSpectrum', spec);
tensHarm       = tensorHarmonicity(pMatB, [], sigma, 'spectrum', spec);

% --- Roughness: the one feature without a batched form ---
% roughness takes one multiset of partials in Hz and depends on their
% absolute frequencies, so transpositions do not share work. Loop over
% the distinct chord rows (transposition included) and map back.
sortedB = sort(pMatB, 2);
[uniqueChords, ~, chordMap] = unique(sortedB, 'rows');
nUnique = size(uniqueChords, 1);
fprintf('  %d trials -> %d distinct chords for the roughness loop.\n\n', ...
    nPairs, nUnique);

uRough   = NaN(nUnique, 1);
refCents = transformAttributes(f0, [], {'hz', 'cents'});
for ui = 1:nUnique
    p = uniqueChords(ui, :);
    p = p(~isnan(p));  % strip NaN padding (if any)
    [pSpec, wSpec] = addSpectra(p(:), [], spec{:});
    fHz = transformAttributes(pSpec + refCents, [], {'cents', 'hz'});
    uRough(ui) = roughness(fHz, wSpec);
end
rough = uRough(chordMap);

% --- Display: one line per distinct chord (its first trial) ---
fprintf('  %-8s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Chord', 'specEnt', 'hMax', 'hEnt', 'tensHarm', 'Rough');
fprintf('  %s\n', repmat('-', 1, 56));

for ui = 1:nUnique
    firstIdx = find(chordMap == ui, 1);
    ci = chordIdx(firstIdx);
    ri = find(roots == rootVals(firstIdx));
    label = sprintf('%s @ %d', chordTypeNames{ci}, roots(ri));

    fprintf('  %-14s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        label, specEnt(firstIdx), hMax(firstIdx), hEnt(firstIdx), ...
        tensHarm(firstIdx), rough(firstIdx));
end

fprintf('\n  (The batched features received all %d rows and computed %d chord types; roughness ran %d times.)\n', ...
    nPairs, nChords, nUnique);

%% === Plot: SPCS heatmaps ===

figure('Name', 'Batch processing demo');
for si = 1:nScales
    subplot(1, nScales, si);
    imagesc(roots, 1:nChords, squeeze(spcsGrid(si, :, :)));
    set(gca, 'YTick', 1:nChords, 'YTickLabel', chordTypeNames);
    xlabel('Root (cents)');
    title(scaleNames{si});
    colorbar;
end

sgtitle('SPCS: chord fit at each root');
colormap(parula);

%% === WORKFLOW 3: A cell of pre-MAETs — batching of a different kind ===
%
%  Everything above batches ROWS: a 2-D pitch matrix whose rows are
%  single multisets over one attribute, deduplicated internally by a
%  canonical key so that 144 rows cost 4 computations. That collapse is
%  possible because the rows are commensurable — same attribute, same
%  geometry, differing only in their values.
%
%  A multi-attribute item has no row to collapse: it is a whole
%  pre-MAET, with its own event count and its own per-attribute
%  geometry. So the multi-attribute analogue of a batch is a CELL, and a
%  cell of pre-MAETs goes wherever a cell of densities goes. The
%  functions build each entry and iterate; nothing is deduplicated,
%  because in general nothing is repeated. What the cell form saves is
%  the calling code — no per-item build, no loop, one call that returns
%  one value per item — not arithmetic.
%
%  The exception that proves the rule is a translation sweep. Its
%  entries DO share one geometry and differ only by an offset, so
%  sweepSimMaet computes the whole sweep in one pass rather than one
%  inner product per offset -- a genuine collapse, and the one place
%  where a multi-attribute batch is cheaper than the loop it replaces.
%  The offsets are what make that possible, and a MATLAB cell cannot
%  carry them alongside the entries, so a swept pre-MAET passed as a
%  cell still loops: the collapse is spelled
%  sweepSimMaet(densX, densY, sweep.offsets), with the offsets taken
%  from translateAttributes' second output.

fprintf('\n=== Workflow 3: A cell of pre-MAETs (batching, other sense) ===\n\n');

% Four two-attribute items: a pitch-class attribute and an onset-time
% attribute. They are NOT commensurable rows — the second has four
% events where the others have three — so no canonical key could
% collapse them.
itemPcs    = {[0 400 700], [0 300 700 1000], [200 500 900], [0 400 700]};
itemOnsets = {[0 1 2],     [0 1 2 3],        [0 1 2],       [0 1 2]};

items = cell(1, numel(itemPcs));
for k = 1:numel(itemPcs)
    pk = {itemPcs{k}, itemOnsets{k}};
    items{k} = packPreMaet(pk, [], flatSpecs(pk, ...
        'name', {'pitch class', 'onset'}, 'sigma', [35 0.25], ...
        'isPer', [true false], 'period', [1200 0]));
end
reference = items{1};

% One call, one value per item. The same call with pre-built densities
% would be identical; the pre-MAETs simply save building them.
sims = simMaet(reference, items, 'verbose', false);
fprintf('  simMaet(reference, {pm1, ..., pm4})\n');
for k = 1:numel(sims)
    fprintf('    item %d: %.4f\n', k, sims{k});
end
fprintf('  (item 1 is the reference; item 4 repeats it.)\n');
fprintf('  Each entry was built and compared in turn — four densities,\n');
fprintf('  four inner products. Nothing collapsed: the items differ in\n');
fprintf('  event count and content, so there is no repeated work to find.\n');

% The sweep is the exception: one geometry, M offsets, computed in one
% pass rather than one inner product per entry. A swept pre-MAET passed
% as a cell is still only the loop -- the one pass needs the offsets,
% and a MATLAB cell cannot carry them, so translateAttributes returns
% them as a second output and sweepSimMaet takes them. (Python attaches
% them to the returned list, so there simMaet picks them up at the call
% site itself.) sweepSimMaet chooses its route by cost and coverage:
% here, the swept attribute (pitch class) being periodic, the orbit
% route.
[pmSweep, sweep] = translateAttributes(reference, {[0 100 200 300], []});
loopSims = simMaet(reference, pmSweep, 'verbose', false);

densRef   = buildMaet(reference, 'verbose', false);
sweepSims = sweepSimMaet(densRef, densRef, sweep.offsets, ...
                               'verbose', false);

fprintf('\n  translateAttributes(reference, {[0 100 200 300], []})\n');
fprintf('    as a cell, one inner product per offset ->');
fprintf(' %.4f', cell2mat(loopSims));
fprintf('\n    as a sweep, sweepSimMaet in one pass    ->');
fprintf(' %.4f', sweepSims);
fprintf('\n');
fprintf('  The two agree to %.1e. Here the entries DO share a geometry\n', ...
        max(abs(cell2mat(loopSims(:))' - sweepSims(:)')));
fprintf('  and differ by a known offset, so the sweep is a genuine\n');
fprintf('  collapse -- the one place where a multi-attribute batch is\n');
fprintf('  cheaper than the loop it replaces.\n');

mptDefaults(prevDefaults);
fprintf('\nDone.\n');
