%DEMO_TRANSLATESWEEP Pre-tensor sliding-comparison sweep with
% translateAttributes and the raw-MA list mode of cosSimExpTens.
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
% query once. The sweep can be specified in either of two equivalent
% forms --- a single A-by-M numeric matrix or a 1-by-G cell with one
% group's sweep per cell --- and Section 3 shows both with a parity
% check.
%
% Compare demo_maetWindowing (post-tensor sliding) and
% demo_windowingReference (reference-point options for
% windowedSimilarity). The pre-tensor route used here returns a strict
% cosine similarity (bounded in [0, 1] for non-negative weights) and
% does not require choosing a window family; the post-tensor route
% returns a magnitude-aware windowed similarity and decouples locality
% from the query's own support.
%
% See also TRANSLATEATTRIBUTES, COSSIMEXPTENS, BUILDEXPTENS,
% WINDOWEDSIMILARITY.

clear; clc;

%% ===================================================================
%  1. Build the reference melody and the query motif (pre-tensor form)
%  ===================================================================

fprintf('=== 1. Pre-tensor inputs ===\n');

% Reference: D-E-F-C-E-G-A at one note per second. The query C-E-G
% appears exactly at times 3, 4, 5.
refMidi   = [62 64 65 60 64 67 69];
refPitch  = convertPitch(refMidi, 'midi', 'cents');
refTime   = 0:6;
refPAttr  = {refPitch, refTime};

% Query: C-E-G triad, 1-second spacing, sweep across both axes.
qryMidi   = [60 64 67];
qryPitch  = convertPitch(qryMidi, 'midi', 'cents');
qryTime   = 0:2;
qryPAttr  = {qryPitch, qryTime};

% Per-group geometry. Two attributes -> two groups (pitch in group 1,
% time in group 2). Pitch is periodic at the octave; time is absolute
% non-periodic.
sigma     = [50, 0.3];        % per-group sigma: cents, seconds
r         = [1, 1];           % single-slot per attribute (K_a = 1)
groups    = [1, 2];           % attribute -> group
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
% Pmesh and Tmesh are used in Section 3 to build the sweep in either
% of the two equivalent offset forms.

fprintf('  pitch grid: %d transpositions over one octave (100-cent steps)\n', ...
        numel(pitchGrid));
fprintf('  time  grid: %d positions from t = %.1f to t = %.1f s\n', ...
        numel(timeGrid), min(timeGrid), max(timeGrid));
fprintf('  total sweep positions: M = %d\n', M);
fprintf('\n');

%% ===================================================================
%  3. Pre-tensor translation: two equivalent offset forms
%  ===================================================================

fprintf('=== 3. translateAttributes (two equivalent offset forms) ===\n');

% Form A: numeric matrix. Rows index attributes, columns index sweep
% positions. With A = 2 singleton groups here, row 1 is the pitch
% attribute and row 2 is the time attribute.
offsetsMat        = zeros(2, M);
offsetsMat(1, :)  = Pmesh(:).';   % pitch shifts (attribute 1)
offsetsMat(2, :)  = Tmesh(:).';   % time  shifts (attribute 2)

qryPAttrSweptMat = translateAttributes(qryPAttr, groups, offsetsMat, ...
                                    isRel, isPer, periods);

% Form B: 1-by-G cell, with one group's sweep per cell. Each entry is
% a 1-by-M row, which the orientation grammar reads as "broadcast
% within group, M-position sweep" --- here that coincides with per-
% attribute because each group is a singleton. Reads naturally as
% "sweep pitch (group 1) by these values; sweep time (group 2) by
% these values".
offsetsCell = {Pmesh(:).', Tmesh(:).'};

qryPAttrSweptCell = translateAttributes(qryPAttr, groups, offsetsCell, ...
                                     isRel, isPer, periods);

% Parity check: the two forms must produce identical translated values.
diffMaxForms = 0;
for m = 1:M
    for a = 1:numel(qryPAttr)
        d = max(abs(qryPAttrSweptMat{m}{a}(:) - qryPAttrSweptCell{m}{a}(:)));
        if d > diffMaxForms, diffMaxForms = d; end
    end
end
fprintf('  matrix form vs cell form: max |diff| = %.2e\n', diffMaxForms);
assert(diffMaxForms == 0, 'Matrix form and cell form disagree.');

% Proceed with the matrix-form output for the downstream computation.
qryPAttrSwept = qryPAttrSweptMat;
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

fprintf('\n=== Demo complete ===\n');
