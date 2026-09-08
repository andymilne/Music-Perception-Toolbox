%% demo_translateSweep.m 
% Pre-tensor sliding-comparison sweep with translateAttributes and the 
% raw-MA list mode of cosSimExpTens.
%
% Scenario: a 3-note motif (C E G) hidden inside a 7-note melody
% (D E F C E G A, one note per second). The motif appears exactly at
% reference times 3, 4, 5. At each (pitch transposition, time shift)
% offset, the query is translated and compared to the un-shifted
% reference with the closed-form cosine similarity of a 2-attribute
% MAET (pitch periodic at the octave; time absolute non-periodic).
% The sweep should peak at (0 cents, 3 s) where the query aligns
% with the embedded C-E-G, and at (1200 cents, 3 s) by octave
% periodicity.
%
% The workflow is two function calls: one to translateAttributes, one to
% cosSimExpTens (raw-MA scalar-vs-list form, with the translated p_attr
% list as one operand and the reference pAttr as the other). The build
% step is internalised: the reference is built once, each translated
% query once. The sweep is specified as a 1-by-A offsets cell, one row
% of M candidate shifts per attribute; Section 3 builds it. Section 7
% shows the same sweep as a single call to sweepCosSimExpTens, which
% never builds the M translated queries at all.
%
% Compare windowedSimilarity (see demo_helixBlend and
% demo_tempoInvariance), which windows the context by event weighting
% before each build --- the window multiplies per-event weights and the
% window axis is then marginalized --- so that locality is decoupled from
% the query's own support. The route used here returns a strict cosine
% similarity (bounded in [0, 1] for non-negative weights) and does not
% require choosing a window family.
%
% See also TRANSLATEATTRIBUTES, COSSIMEXPTENS, SWEEPCOSSIMEXPTENS,
% BUILDEXPTENS, WINDOWEDSIMILARITY.

clear; clc;

%% ===================================================================
%  1. Build the reference melody and the query motif (pre-tensor form)
%  ===================================================================

fprintf('=== 1. Pre-tensor inputs ===\n');

% Reference: D-E-F-C-E-G-A at one note per second. The query C-E-G
% appears exactly at times 3, 4, 5.
refMidi   = [62 64 65 60 64 67 69];
refPitch  = transformAttributes(refMidi, [], {'midi', 'cents'});
refTime   = 0:6;
refPAttr  = {refPitch, refTime};

% Query: C-E-G triad, 1-second spacing, sweep across both axes.
qryMidi   = [60 64 67];
qryPitch  = transformAttributes(qryMidi, [], {'midi', 'cents'});
qryTime   = 0:2;
qryPAttr  = {qryPitch, qryTime};

% Per-attribute geometry: pitch (attribute 1) is periodic at the
% octave; time (attribute 2) is absolute non-periodic.
sigma     = [50, 0.3];        % per-attribute sigma: cents, seconds
r         = [1, 1];           % single-value per attribute (K_a = 1)
isRel     = [false, false];
isPer     = [true,  false];
periods   = [1200,  0];

fprintf('  reference: D-E-F-C-E-G-A, one note per second\n');
fprintf('  query    : C-E-G triad, 1-second spacing\n');
fprintf('  (the motif appears exactly at reference times 3, 4, 5)\n');
fprintf('  sigma    : %.0f cents (pitch) / %.2f s (time)\n', ...
        sigma(1), sigma(2));
fprintf('\n');

%% ===================================================================
%  2. Construct the (pitch, time) offset sweep
%  ===================================================================

fprintf('=== 2. Offset sweep grid ===\n');

% Pitch offsets: 0 - 1200 cents in 100-cent steps (one octave). The
% expected peak at pitch shift 0 is also visible at 1200 cents because
% the pitch group is octave-periodic.
pitchGrid = 0:100:1200;
% Time offsets: -1 to 5 seconds in 0.25-second steps.
timeGrid  = -1:0.25:5;

[Pmesh, Tmesh] = meshgrid(pitchGrid, timeGrid);
M = numel(Pmesh);
% Pmesh and Tmesh are flattened in Section 3 into the per-attribute
% rows of the offsets cell.

fprintf('  pitch grid: %d transpositions over one octave (100-cent steps)\n', ...
        numel(pitchGrid));
fprintf('  time  grid: %d positions from t = %.1f to t = %.1f s\n', ...
        numel(timeGrid), min(timeGrid), max(timeGrid));
fprintf('  total sweep positions: M = %d\n', M);
fprintf('\n');

%% ===================================================================
%  3. Pre-tensor translation: build the swept query
%  ===================================================================

fprintf('=== 3. translateAttributes (offset sweep) ===\n');

% offsets is a 1-by-A cell, one entry per attribute. Each entry here is
% a 1-by-M row, which the orientation grammar reads as a per-sweep
% global shift: M candidate offsets broadcast across the attribute's
% values (trivial here, as each attribute is single-value, K_a = 1). The
% M sweep columns are shared across attributes, so column m of every
% entry together defines the m-th translated copy. Reads naturally as
% "sweep pitch by these values; sweep time by these values".
offsetsCell   = {Pmesh(:).', Tmesh(:).'};
[pmSwept, sweep] = translateAttributes(qryPAttr, [], offsetsCell);
qryPAttrSwept = pmSwept.pAttr;
% The fourth output records the per-attribute offsets (A x M) for
% sweepCosSimExpTens; see Section 7.

fprintf('  qryPAttrSwept: %s, length %d\n', class(qryPAttrSwept), ...
        numel(qryPAttrSwept));
fprintf('  each entry is a 1-by-%d cell of K_a-by-N value matrices\n', ...
        numel(qryPAttr));
fprintf('\n');

%% ===================================================================
%  4. Raw-MA scalar-vs-list cosine similarity: one call
%  ===================================================================

fprintf('=== 4. cosSimExpTens (raw-MA list mode) ===\n');

sCells = cosSimExpTens(refPAttr, [], qryPAttrSwept, [], ...
                        sigma, r, isRel, isPer, periods, ...
                        'verbose', false);
S      = cell2mat(sCells);             % 1-by-M
S      = reshape(S, size(Pmesh));      % size = [numel(timeGrid), numel(pitchGrid)]

% Locate the peak.
[sMax, iLin] = max(S(:));
[iT, iP]     = ind2sub(size(S), iLin);
fprintf('  cosine similarity profile: %d x %d (time x pitch)\n', ...
        size(S, 1), size(S, 2));
fprintf('  max similarity %.4f at pitch shift %.0f c, time shift %.2f s\n', ...
        sMax, pitchGrid(iP), timeGrid(iT));
fprintf('  (expected: 0 cents, 3.00 s --- the embedded C-E-G)\n');
fprintf('\n');

%% ===================================================================
%  5. Visualise the sweep
%  ===================================================================

fprintf('=== 5. Plot ===\n');

fig = figure('Name', 'demo\_translateSweep: pre-tensor sliding cosine', ...
             'Position', [100 100 900 600], 'Color', 'w');

imagesc(pitchGrid, timeGrid, S);
axis xy;
colorbar;
xlabel('Pitch transposition (cents)');
ylabel('Time shift (s)');
title({'Pre-tensor sliding-comparison: cosine similarity', ...
       'Reference: D-E-F-C-E-G-A; query: C-E-G'});
set(gca, 'XTick', 0:200:1200, 'YTick', -1:1:5);
hold on;

% Mark the expected peak positions: query aligns with the embedded
% C-E-G at (0 c, 3 s). Octave periodicity reproduces the peak at
% (1200 c, 3 s).
plot([0 1200], [3 3], 'rx', 'MarkerSize', 12, 'LineWidth', 1.5);
text(40,   3.4, 'C-E-G match', 'Color', 'r', ...
     'FontSize', 9, 'BackgroundColor', [1 1 1 0.7]);
text(1100, 3.4, 'octave', 'Color', 'r', ...
     'FontSize', 9, 'BackgroundColor', [1 1 1 0.7]);

fprintf('  Figure shows the cosine-similarity surface as a function of\n');
fprintf('  pitch transposition and time shift. Red x marks the\n');
fprintf('  expected peaks at (0 c, 3 s) and (1200 c, 3 s), where the\n');
fprintf('  query aligns with the embedded C-E-G in the reference;\n');
fprintf('  octave-pitch periodicity makes the two peaks identical.\n');
fprintf('\n');

%% ===================================================================
%  6. Equivalent explicit build loop, for transparency
%  ===================================================================

fprintf('=== 6. Equivalent explicit build loop ===\n');
fprintf('  This is what the raw-MA list mode does internally; here it\n');
fprintf('  is spelled out so the relationship between translateAttributes,\n');
fprintf('  buildExpTens, and cosSimExpTens is transparent.\n\n');

showPreMaet(refPAttr, [], [], 'names', {'pitch', 'time'}, ...
    'sigma', sigma, 'isRel', isRel, 'isPer', isPer, 'period', periods);
showPreMaet(qryPAttrSwept{1}, [], [], 'names', {'pitch', 'time'}, ...
    'sigma', sigma, 'isRel', isRel, 'isPer', isPer, 'period', periods);
fprintf('\n');

densRef = buildExpTens(refPAttr, [], sigma, r, ...
                       isRel, isPer, periods, 'verbose', false);
S_manual = zeros(1, M);
for m = 1:M
    densQ = buildExpTens(qryPAttrSwept{m}, [], sigma, r, ...
                         isRel, isPer, periods, 'verbose', false);
    S_manual(m) = cosSimExpTens(densRef, densQ, 'verbose', false);
end
S_manual = reshape(S_manual, size(Pmesh));

discrepancy = max(abs(S(:) - S_manual(:)));
fprintf('  max |S_raw - S_manual| = %.2e (floating-point parity)\n', ...
        discrepancy);
assert(discrepancy < 1e-12, 'Raw-MA list mode disagrees with manual build loop.');

%% ===================================================================
%  7. The same sweep without building M queries: sweepCosSimExpTens
%  ===================================================================

fprintf('\n=== 7. sweepCosSimExpTens (one call, no translated copies) ===\n');
fprintf('  A uniform translation of the query enters the inner product only\n');
fprintf('  through the offset, so the whole sweep is one pass over the tuple\n');
fprintf('  pairs and then one evaluation per offset. The pitch attribute is\n');
fprintf('  periodic, which the mixture route refuses; under ''method'', ''auto''\n');
fprintf('  the orbit route carries the sweep instead (the wrapped kernel\n');
fprintf('  absorbs the periodicity), so the call is the same either way.\n');

densQry = buildExpTens(qryPAttr, [], sigma, r, ...
                       isRel, isPer, periods, 'verbose', false);
S_sweep = sweepCosSimExpTens(densRef, densQry, sweep.offsets, 'verbose', false);
discrepancySweep = max(abs(S(:).' - S_sweep(:).'));
fprintf('  max |S_raw - S_sweep| = %.2e\n', discrepancySweep);
assert(discrepancySweep < 1e-8, 'sweepCosSimExpTens disagrees with the per-offset route.');

fprintf('\n=== Demo complete ===\n');
