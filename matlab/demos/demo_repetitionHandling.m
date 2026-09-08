%% demo_repetitionHandling.m
%  Handling repeated pitches under interval-scale and tempo invariance.
%
%  A melodic contour can be matched up to a uniform scaling of its
%  intervals -- every step multiplied by the same factor -- by carrying
%  each pitch step as two attributes: its sign, and the logarithm of
%  its magnitude. A scaling then leaves every sign fixed and adds the
%  same constant, log(a), to every log-magnitude, so reading the bound
%  log-magnitude tuple relative to a common shift ('relOuter' = true)
%  makes the match exact, just as log inter-onset intervals read
%  relative make a match exact across tempo (demo_tempoInvariance).
%
%  A repeated pitch breaks this: its step is 0, and 0 has no logarithm.
%  Something must be done with repetitions before the decomposition,
%  and the choice decides what a repetition is worth in the comparison.
%  This demo walks through three treatments of the same melody:
%
%    excise    Drop each zero-step event (its step AND its inter-onset
%              interval). A repetition leaves no trace: the melody
%              becomes indistinguishable from the same melody with its
%              repetitions removed and the rhythm closed up.
%    prolong   Gather each run of equal consecutive pitches into one
%              event at the run's first onset, before differencing.
%              The repetition survives as duration: the gathered
%              event's next inter-onset interval spans the whole run.
%              The cost is an identification: a re-articulated note and
%              a single held note of the same length produce the same
%              events.
%    count     Prolong, and add a per-event attribute holding the
%              number of onsets gathered into the event.
%              Re-articulation now survives as such, and its sigma
%              grades how sharply it is distinguished -- wide, and the
%              comparison slides back to the prolong treatment's
%              identification.
%
%  Each treatment is run twice: once with inter-onset intervals in
%  beats (tempo-sensitive -- a faster statement does not match), and
%  once with log inter-onset intervals read relative (tempo-invariant
%  -- it does). The second run exposes a confound the first hides:
%  where repetition is uniform (every note repeated equally), closing
%  up the repetitions IS a tempo change, and only the count attribute
%  keeps the two apart.
%
%  Sections:
%    1. Material            The melody and five comparison variants.
%    2. The three treatments, shown as events (the reference melody).
%    3. Inter-onset intervals in beats: which variants match ref under
%       which treatment.
%    4. Log inter-onset intervals read relative: the same table, with
%       tempo quotiented out.
%    5. The count attribute's sigma, from sharp to broad.
%    6. Uniform repetition: closing up equals a tempo change, and the
%       count attribute is what keeps them apart.
%
%  Exact invariances only ('relOuter' flags); the graded counterparts
%  (the sdShift ridge of intervalKernelCov, tending to rel in the
%  limit) are the subject of demo_tempoInvariance.

clear; close all;

% Keep the dispatcher's per-call announcements out of the printed
% tables (showHints gates only those; the one-time truncation notice
% is not gated). The controls are covered in demo_dispatchAndKernelControls.
prevDefaults = mptDefaults('showHints', false);

% Kernel widths, one per attribute. The sign attribute is two points a
% unit apart (+1/2 and -1/2), so its narrow sigma makes sign an exact
% comparison; log-magnitude and log-IOI sigmas are in natural-log
% units; the IOI-in-beats sigma is in beats; the count sigma is in
% onsets and is varied in Section 5.
SIGMA_SIGN = 0.05;
SIGMA_LOGMAG = 0.15;
SIGMA_IOI_BEATS = 0.25;
SIGMA_LOGIOI = 0.15;
SIGMA_COUNT = 0.25;

%% ===== 1. Material =====

% The reference melody: the Ode to Joy opening, isochronous, one onset
% per beat. Five of its fifteen notes are repetitions of the note
% before (E E, G G, C C, E E, D D), interleaved with unrepeated notes,
% so the repetition pattern is non-uniform -- Section 6 shows why that
% matters.
pitchesRef = [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62];
onsetsRef = 0:14;

% The five comparison variants, each a {pitches, onsets} pair:
%   held       The gathered reference played as sustained notes: same
%              pitches at the same onsets, but each formerly repeated
%              note now a single held note -- no re-articulations.
%   norep      The repetitions removed and the rhythm closed up: the
%              ten distinct pitches, one per beat.
%   augmented  Every pitch step doubled (contour preserved), rhythm as
%              the reference. Repetitions remain repetitions: a zero
%              step doubles to zero.
%   faster     The reference at double speed.
%   aug+faster Both at once.
[gp, go, ~] = gatherRepetitions(pitchesRef, onsetsRef);
stepsRef = diff(pitchesRef);
pitchesAug = pitchesRef(1) + [0, cumsum(2 * stepsRef)];
variantNames = {'held', 'norep', 'augmented', 'faster', 'aug+faster'};
variantP = {gp, gp, pitchesAug, pitchesRef, pitchesAug};
variantT = {go, 0:(numel(gp) - 1), onsetsRef, 0.5 * onsetsRef, ...
            0.5 * onsetsRef};

fprintf('\n=== 1. Material ===\n\n');
fprintf('  reference : Ode to Joy opening, 15 notes, one per beat;\n');
fprintf('              5 notes are repetitions of the note before.\n');
for v = 1:numel(variantNames)
    fprintf('  %-10s: %d onsets\n', variantNames{v}, numel(variantP{v}));
end
fprintf('\n');

%% ===== 2. The three treatments, shown as events =====

treatments = {'excise', 'prolong', 'count'};

fprintf('=== 2. The reference melody under each treatment ===\n\n');
for t = 1:numel(treatments)
    ev = makeEvents(pitchesRef, onsetsRef, treatments{t});
    fprintf('  %s: %d events\n', treatments{t}, numel(ev.steps));
    fprintf('    step sign  :');
    for i = 1:numel(ev.steps)
        if ev.steps(i) > 0, fprintf('  +'); else, fprintf('  -'); end
    end
    fprintf('\n    |step|     :');
    fprintf('  %.0f', abs(ev.steps));
    fprintf('\n    IOI (beats):');
    fprintf('  %.0f', ev.ioi);
    fprintf('\n');
    if isfield(ev, 'count')
        fprintf('    onsets     :');
        fprintf('  %.0f', ev.count);
        fprintf('\n');
    end
    fprintf('\n');
end
fprintf('  The excised and prolonged step sequences coincide; they part\n');
fprintf('  on the intervals. Excision keeps each surviving event''s own\n');
fprintf('  1-beat interval, so the repetitions'' beats are gone; the\n');
fprintf('  prolonged events'' intervals span the gathered runs, so those\n');
fprintf('  beats survive as duration. The count row records what\n');
fprintf('  prolongation alone forgets: how many onsets each event held.\n\n');

%% ===== 3 & 4. Which variants match the reference =====

fprintf('=== 3. Inter-onset intervals in beats ===\n\n');
fprintf('  Cosine similarity of each variant to the reference. The\n');
fprintf('  log-magnitude tuple is read relative throughout, so the\n');
fprintf('  augmented variant -- every step doubled, rhythm unchanged --\n');
fprintf('  matches exactly under every treatment. The intervals are in\n');
fprintf('  beats, so the faster variants do not. At these narrow kernel\n');
fprintf('  widths each entry reads as identified (1.000) or separated\n');
fprintf('  (near 0); widening an attribute''s sigma grades its\n');
fprintf('  separations, as Section 5 does for the count.\n\n');
similarityTable(false, treatments, pitchesRef, onsetsRef, ...
    variantNames, variantP, variantT, SIGMA_SIGN, SIGMA_LOGMAG, ...
    SIGMA_IOI_BEATS, SIGMA_LOGIOI, SIGMA_COUNT);
fprintf('  Each treatment commits to one identification, visible in its\n');
fprintf('  column''s 1.000: excise cannot tell the reference from norep\n');
fprintf('  (the repetitions leave no trace), prolong cannot tell it from\n');
fprintf('  held (re-articulation and prolongation coincide), and count\n');
fprintf('  distinguishes all four.\n\n');

fprintf('=== 4. Log inter-onset intervals, read relative ===\n\n');
fprintf('  The same comparisons with the intervals taken to logarithms\n');
fprintf('  and the bound tuple read relative: a tempo change is a common\n');
fprintf('  shift of the log intervals, so the faster variants now match\n');
fprintf('  exactly -- including aug+faster, scaled in pitch and time at\n');
fprintf('  once.\n\n');
similarityTable(true, treatments, pitchesRef, onsetsRef, ...
    variantNames, variantP, variantT, SIGMA_SIGN, SIGMA_LOGMAG, ...
    SIGMA_IOI_BEATS, SIGMA_LOGIOI, SIGMA_COUNT);
fprintf('  The treatments'' identifications survive the tempo quotient\n');
fprintf('  here because the reference''s repetitions are non-uniform: its\n');
fprintf('  gathered intervals (2 1 2 1 1 1 2 1 2) are not a common\n');
fprintf('  scaling of norep''s (1 1 ... 1), so prolongation still\n');
fprintf('  separates them. Section 6 shows the uniform case, where it\n');
fprintf('  cannot.\n\n');

%% ===== 5. The count attribute's sigma =====

fprintf('=== 5. Grading the count attribute ===\n\n');
fprintf('  Under the count treatment, reference vs held differ only on\n');
fprintf('  the count attribute (2 vs 1 at the five gathered events). Its\n');
fprintf('  sigma sets how much that difference costs: narrow, the two\n');
fprintf('  are far apart; broad, the counts blur together and the\n');
fprintf('  comparison returns to the prolong treatment''s identification\n');
fprintf('  of re-articulated with held.\n\n');
evRef = makeEvents(pitchesRef, onsetsRef, 'count');
evHeld = makeEvents(variantP{1}, variantT{1}, 'count');
for sc = [0.25, 0.75, 1.5, 3.0]
    dRef = buildDensity(evRef, true, SIGMA_SIGN, SIGMA_LOGMAG, ...
        SIGMA_IOI_BEATS, SIGMA_LOGIOI, sc);
    dHeld = buildDensity(evHeld, true, SIGMA_SIGN, SIGMA_LOGMAG, ...
        SIGMA_IOI_BEATS, SIGMA_LOGIOI, sc);
    s = cosSimExpTens(dRef, dHeld, 'verbose', false);
    fprintf('    count sigma = %.2f: similarity = %.3f\n', sc, s);
end
fprintf('\n');

%% ===== 6. Uniform repetition: closing up equals a tempo change =====

fprintf('=== 6. Uniform repetition ===\n\n');
fprintf('  A figure whose every note is repeated: C C G G A A G G, one\n');
fprintf('  onset per beat, against the same figure with the repetitions\n');
fprintf('  removed and closed up: C G A G. Gathering the first gives the\n');
fprintf('  pitches of the second at intervals (2 2 2) against (1 1 1) --\n');
fprintf('  exactly a common factor, so once tempo is quotiented out the\n');
fprintf('  prolong treatment cannot separate them: removing uniform\n');
fprintf('  repetition IS a tempo change. The counts (2 2 2) against\n');
fprintf('  (1 1 1) are untouched by either quotient, so the count\n');
fprintf('  treatment can.\n\n');
pUnif = [60, 60, 67, 67, 69, 69, 67, 67];
tUnif = 0:7;
pUnorep = [60, 67, 69, 67];
tUnorep = 0:3;
for t = 2:3   % prolong, count
    d1 = buildDensity(makeEvents(pUnif, tUnif, treatments{t}), true, ...
        SIGMA_SIGN, SIGMA_LOGMAG, SIGMA_IOI_BEATS, SIGMA_LOGIOI, ...
        SIGMA_COUNT);
    d2 = buildDensity(makeEvents(pUnorep, tUnorep, treatments{t}), true, ...
        SIGMA_SIGN, SIGMA_LOGMAG, SIGMA_IOI_BEATS, SIGMA_LOGIOI, ...
        SIGMA_COUNT);
    s = cosSimExpTens(d1, d2, 'verbose', false);
    fprintf('    %-8s: doubled figure vs closed-up figure = %.3f\n', ...
        treatments{t}, s);
end
fprintf('\n');
fprintf('  With intervals in beats the two are already distinct under\n');
fprintf('  every treatment; the confound is a price of tempo invariance,\n');
fprintf('  and the count attribute -- dimensionless, so invariant to\n');
fprintf('  both scalings for free -- is what pays it off.\n');

mptDefaults(prevDefaults);

%% ---- Local helpers (script-scope; must follow all executable code) ----

function [pOut, tOut, counts] = gatherRepetitions(pitches, onsets)
%GATHERREPETITIONS  Pool each run of equal consecutive pitches into one
%   event. Returns the run-start pitches, the run-start onsets, and the
%   run lengths (how many onsets each pooled event gathers).
    starts = [true, abs(diff(pitches)) > 1e-9];
    idx = find(starts);
    pOut = pitches(idx);
    tOut = onsets(idx);
    counts = diff([idx, numel(pitches) + 1]);
end

function ev = makeEvents(pitches, onsets, treatment)
%MAKEEVENTS  Differenced events for one variant under one treatment.
%   Returns a struct with the per-event arrays: sign (+1/2 or -1/2),
%   logmag (log of the absolute pitch step), ioi (inter-onset interval
%   in beats), and, for the count treatment, count (onsets gathered
%   into the event completing the step).
    switch treatment
        case {'prolong', 'count'}
            [pitches, onsets, counts] = gatherRepetitions(pitches, onsets);
            [pDiff, ~, ~] = unpackPreMaet(differenceEvents( ...
                {pitches, onsets, counts}, [], [1, 1, 0]));
            steps = pDiff{1};
            iois = pDiff{2};
            counts = pDiff{3};
        case 'excise'
            [pDiff, ~, ~] = unpackPreMaet(differenceEvents({pitches, onsets}, [], [1, 1]));
            steps = pDiff{1};
            iois = pDiff{2};
            keep = abs(steps) > 1e-9;
            steps = steps(keep);
            iois = iois(keep);
            counts = [];
        otherwise
            error('demo_repetitionHandling:unknownTreatment', ...
                'unknown treatment ''%s''', treatment);
    end
    ev = struct();
    ev.steps = steps;
    ev.ioi = iois;
    if strcmp(treatment, 'count')
        ev.count = counts;
    end
end

function dens = buildDensity(ev, logIoi, sigmaSign, sigmaLogmag, ...
    sigmaIoiBeats, sigmaLogioi, sigmaCount, showInput)
%BUILDDENSITY  One bound super-event: the whole event sequence as one
%   tuple. Attributes: log step magnitude, sign, inter-onset interval,
%   and (when present) count. One transformAttributes call takes the log
%   of the signed step and, with 'sign' true, inserts the sign attribute
%   right after it at the 2-point simplex's vertices, {-1/2, 0, +1/2};
%   the same call takes the log of the
%   intervals where logIoi. The log-magnitude tuple is read relative
%   ('relOuter'), quotienting a common shift -- a uniform scaling of
%   the pitch steps. With logIoi, the intervals are read relative too,
%   quotienting a tempo change; in beats they are read absolute, so
%   tempo differences count.
    if logIoi
        ioiTransform = 'log';
        sigmaIoi = sigmaLogioi;
    else
        ioiTransform = [];
        sigmaIoi = sigmaIoiBeats;
    end
    pmStep = transformAttributes({ev.steps, ev.ioi}, [], ...
        {'log', ioiTransform}, 'sign', [true, false]);
    pAttr = pmStep.pAttr;                  % {log magnitude, sign, ioi}
    rel = [true, false, logIoi];
    sig = [sigmaLogmag, sigmaSign, sigmaIoi];
    if isfield(ev, 'count')
        pAttr{end + 1} = ev.count;
        rel(end + 1) = false;
        sig(end + 1) = sigmaCount;
    end
    L = numel(ev.steps);
    [pB, wB, spB] = unpackPreMaet(bindEvents(pAttr, [], L, 'relOuter', rel));
    if nargin >= 8 && showInput
        if logIoi
            ioiName = 'log IOI';
        else
            ioiName = 'IOI';
        end
        names = {'log magnitude', 'sign', ioiName};
        if numel(sig) > 3
            names{end + 1} = 'count';
        end
        showPreMaet(pB, wB, spB, 'names', names, 'sigma', sig, ...
            'isPer', false(1, numel(sig)), 'maxElements', 6);
        fprintf('\n');
    end
    dens = buildExpTens(pB, wB, 'specs', spB, 'sigma', sig, ...
        'isPer', false(1, numel(sig)), 'period', zeros(1, numel(sig)), ...
        'verbose', false);
end

function similarityTable(logIoi, treatments, pitchesRef, onsetsRef, ...
    variantNames, variantP, variantT, sigmaSign, sigmaLogmag, ...
    sigmaIoiBeats, sigmaLogioi, sigmaCount)
%SIMILARITYTABLE  Cosine similarity of each variant to the reference,
%   one column per treatment.
    nT = numel(treatments);
    refDens = cell(1, nT);
    for t = 1:nT
        refDens{t} = buildDensity( ...
            makeEvents(pitchesRef, onsetsRef, treatments{t}), logIoi, ...
            sigmaSign, sigmaLogmag, sigmaIoiBeats, sigmaLogioi, ...
            sigmaCount, t == 1);
    end
    fprintf('  %-12s', 'variant');
    fprintf('%10s', treatments{:});
    fprintf('\n  %s\n', repmat('-', 1, 12 + 10 * nT));
    for v = 1:numel(variantNames)
        fprintf('  %-12s', variantNames{v});
        for t = 1:nT
            d = buildDensity( ...
                makeEvents(variantP{v}, variantT{v}, treatments{t}), ...
                logIoi, sigmaSign, sigmaLogmag, sigmaIoiBeats, ...
                sigmaLogioi, sigmaCount);
            s = cosSimExpTens(refDens{t}, d, 'verbose', false);
            fprintf('%10.3f', s);
        end
        fprintf('\n');
    end
    fprintf('\n');
end
