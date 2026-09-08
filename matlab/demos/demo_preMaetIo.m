%% demo_preMaetIo.m
% Showing, exporting, and importing a pre-MAET.
%
% The pre-MAET is the framework's interface: the event sequence, its
% per-event-attribute element multisets, and the per-attribute parameters
% (Milne 2026, Def. 2.6). Everything a MAET computes follows from it
% mechanically, so being able to read one, write one, and hand one to a
% colleague as a spreadsheet is the point of this demo.
%
% Four renderings of the same object:
%
%   1. markdown, for a terminal;
%   2. LaTeX, in the article's own markup, for a manuscript table;
%   3. CSV, written out for Excel or Numbers;
%   4. CSV, read back in --- byte for byte the same object.
%
% The cell notation is the article's throughout: braces for an unordered
% multiset, parentheses for an ordered one, brackets within brackets for a
% nested attribute, and 60^(0.6) for a weighted value. showPreMaet writes
% that notation and readPreMaet reads it, so the two are inverse and a
% pre-MAET survives a round trip through a spreadsheet.
%
%   Sections
%       1. A pre-MAET by hand, shown four ways.
%       2. Round trip: write, read, and compare.
%       3. Building straight from the file.
%       4. Overriding a parameter the pre-MAET carries.
%       5. A nested attribute, with weights.
%       6. NA: a parameter a preprocessing step could not carry forward.
%       7. An anisotropic kernel, carried as its three generating scalars.
%       8. A pre-MAET from a score, which supplies what a score determines.
%
% The Python mirror is demos/demo_pre_maet_io.py.
%
% See also SHOWPREMAET, READPREMAET, WRITEPREMAET, BUILDEXPTENS.

clear; clc;
outDir = tempname; mkdir(outDir);

%% ===================================================================
%  1. One pre-MAET, four renderings
%  ===================================================================

fprintf('=== 1. One pre-MAET, four renderings ===\n\n');

% Three chords of a cadence, with their onsets. Pitch is an unordered
% multiset read at r = 2 (shared pitch pairs) and periodic at the octave;
% time is a single value per event on an unbounded axis.
pAttr = { [62 55 60; 65 59 64; 69 62 67; 72 65 NaN], ...
          [0 1 2] };
specs = { struct('name', 'pitch', 'r', 2, 'rel', false, 'sym', true, ...
                 'sigma', 0.15, 'isPer', true, 'period', 12), ...
          struct('name', 'onset', 'r', 1, 'rel', false, 'sym', true, ...
                 'sigma', 0.1, 'isPer', false, 'period', 0) };

% preMaet holds the three parts in one variable, which every function
% below then takes whole. The specs carry the kernel geometry, so nothing
% further is needed here: the pre-MAET is complete as it stands.
pm = preMaet(pAttr, [], specs);

fprintf('  (a) markdown\n\n');
showPreMaet(pm);

fprintf('\n  (b) LaTeX\n\n');
showPreMaet(pm, 'format', 'latex', ...
    'caption', 'A cadence as a pre-MAET.', 'label', 'tab:cadence');

fprintf('\n  (c) CSV\n\n');
fprintf('%s', showPreMaet(pm, 'format', 'csv', 'verbose', false));

%% ===================================================================
%  2. Round trip through a spreadsheet
%  ===================================================================

fprintf('\n=== 2. Round trip through a spreadsheet ===\n\n');

pathCsv = fullfile(outDir, 'cadence.csv');
written = writePreMaet(pathCsv, pm);
fprintf('  written to cadence.csv\n');

pmBack = readPreMaet(pathCsv);
again = writePreMaet([], pmBack);

valsOk = true;
for a = 1:2
    valsOk = valsOk && isequaln(pAttr{a}, pmBack.pAttr{a});
end
fprintf('  values identical : %d\n', valsOk);
fprintf('  file identical   : %d\n', strcmp(again, written));
fprintf(['  (the NaN pad of the three-note chord survives as a shorter ' ...
         'cell.)\n\n']);

%% ===================================================================
%  3. From file to density, with nothing supplied
%  ===================================================================

fprintf('=== 3. From file to density, with nothing supplied ===\n\n');

dens = buildExpTens(pmBack, 'verbose', false);
fprintf('  buildExpTens(pm)                    ->  dim %d\n', dens.dim);
fprintf('  self-similarity                     ->  %.4f\n', ...
    cosSimExpTens(dens, dens, 'verbose', false));
fprintf('  No sigma, isPer or period passed: the file carried them.\n\n');

%% ===================================================================
%  4. Overriding what the pre-MAET carries
%  ===================================================================

fprintf('=== 4. Overriding what the pre-MAET carries ===\n\n');

% A parameter given at the call wins over the one in the spec, for every
% attribute and without comment: holding a baseline in the pre-MAET and
% sweeping a width past it is the ordinary idiom, so a disagreement is
% intent rather than error. All six per-attribute parameters resolve this
% way -- sigma, isPer and period, and r, rel and sym -- so a sweep over
% any of them is one call per value.
%
% showPreMaet reads the override too, so the table states what the build
% will use rather than what the file said.
showPreMaet(pm, 'sigma', [0.6 0.1], 'title', '  with sigma = [0.6 0.1]:');

% What the width buys is visible against a semitone shift: the wider the
% pitch kernel, the more nearly the shifted cadence matches the original.
pmUp = preMaet({pm.pAttr{1} + 1, pm.pAttr{2}}, [], pm.specs);

fprintf('\n');
sigmasPitch = [0.05 0.15 0.6 2];
for k = 1:numel(sigmasPitch)
    kw = {'sigma', [sigmasPitch(k) 0.1], 'verbose', false};
    sim = cosSimExpTens(buildExpTens(pm, kw{:}), ...
                        buildExpTens(pmUp, kw{:}), 'verbose', false);
    fprintf('  sigma_pitch = %.2f  ->  vs a semitone up: %.4f\n', ...
        sigmasPitch(k), sim);
end

% The sweep reads the pre-MAET and never writes to it, so the baseline is
% still there afterwards.
fprintf('\n  pm''s own sigma is still %.2f.\n', pm.specs{1}.sigma);
fprintf(['  A nested attribute''s r, rel and sym are per-level, so an ' ...
         'override\n  is refused there and the spec is the place to ' ...
         'change them.\n\n']);

%% ===================================================================
%  5. A nested attribute, with weights
%  ===================================================================

fprintf('=== 5. A nested attribute, with weights ===\n\n');

% Bind the three chords into one super-event: an ordered run of unordered
% chords, the shape the cadence prototypes of the article use. The kernel
% geometry crosses to the nested spec intact.
pmB = bindEvents(pm, [3 3]);
showPreMaet(pmB);
fprintf('\n');
fprintf('%s', showPreMaet(pmB, 'format', 'csv', 'verbose', false));
fprintf(['\n  The brackets are the level structure: readPreMaet rebuilds ' ...
         'the\n  tags from them, so a file never has to write them ' ...
         'down.\n\n']);

%% ===================================================================
%  6. NA, where a step could not carry a parameter
%  ===================================================================

fprintf('=== 6. NA, where a step could not carry a parameter ===\n\n');

% A log is non-linear. A width is still meaningful on the log axis -- it
% expresses a ratio rather than a difference -- but the local scaling
% varies across the range, so no single value is the image of the old
% sigma. NA marks the absence of a canonical choice, and the analyst
% supplies the width the new units call for.
pmLog = transformAttributes(preMaet({pAttr{2} + 1}, [], specs(2)), ...
    {'log'});
showPreMaet(pmLog);
try
    buildExpTens(pmLog, 'verbose', false);
catch err
    fprintf('\n  build refuses it: %s\n', err.message);
end
fprintf('\n  Supplying a width resolves it, the analyst having chosen one:\n');
dLog = buildExpTens(pmLog, 'sigma', 0.05, ...
    'verbose', false);
fprintf('    dim %d\n\n', dLog.dim);

%% ===================================================================
%  7. A kernel covariance, as three scalars
%  ===================================================================

fprintf('=== 7. A kernel covariance, as three scalars ===\n\n');

% A covariance is a matrix, but the covariance an analysis wants is
% generated by three numbers, so those are what the file carries.
covCsv = ['name,sigma,r,rel,per,P,sym,n = 1,n = 2' sprintf('\n') ...
    'trigram,"cov(sd_position=0.2, sd_interval=0.3, sd_shift=0.5)",' ...
    '3,0,0,,0,"(60, 62, 64)","(62, 64, 65)"' sprintf('\n')];
pmC = readPreMaet(covCsv);
fprintf('  read back as a matrix:\n');
disp(round(pmC.specs{1}.sigma * 10000) / 10000);
showPreMaet(pmC);
dC = buildExpTens(pmC, 'verbose', false);
fprintf('\n  builds with a kernel covariance: %d\n', ~isempty(dC.kernelCov{1}));
fprintf('  and writes back unchanged      : %d\n\n', ...
    strcmp(writePreMaet([], pmC), covCsv));

%% ===================================================================
%  8. From a score
%  ===================================================================

fprintf('=== 8. From a score ===\n\n');

% The chorale the JMM demos analyse, which ships with the demos: a demo
% should not reach into the test tree for its data.
here = fileparts(mfilename('fullpath'));
score = fullfile(here, 'jmm', 'data', 'bwv347.musicxml');
if exist(score, 'file')
    pmS = preMaetFromScore(score, ...
        'attributes', {'pitch', 'onset'}, 'chords', 'bind');
    % A score determines periodicity and not kernel widths, so sigma is
    % left for the analyst; the table shows what is still missing.
    showPreMaet(pmS, 'maxEvents', 5, 'maxElements', 4);
    try
        buildExpTens(pmS, 'verbose', false);
    catch err
        fprintf('\n  %s\n', err.message);
    end
    sigmas = [0.15, 0.1];
    for a = 1:numel(pmS.specs)
        pmS.specs{a}.sigma = sigmas(a);
    end
    outCsv = fullfile(outDir, 'from_score.csv');
    writePreMaet(outCsv, pmS);
    fprintf('\n  widths chosen and exported to from_score.csv\n');
    fprintf(['  -- the analysis is now a spreadsheet a colleague can ' ...
             'edit.\n']);
else
    fprintf('  (score fixture not found; skipping)\n');
end

fprintf('\nFiles written to %s\n', outDir);
