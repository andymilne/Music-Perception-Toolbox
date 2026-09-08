%% demo_tempoInvariance.m
%  Anisotropic kernels for tempo tolerance and tempo invariance:
%  searching for a rhythmic motif in an onset stream with
%  intervalKernelCov and windowedSimilarity.
%
%  A matrix-valued kernel covariance (accepted wherever sigma is, on an
%  ordered, absolute, non-periodic, non-nested attribute whose tuple is
%  its whole multiset, r == K) lets one Gaussian kernel express several
%  independent sources of perceptual tolerance at once. The constructor
%  intervalKernelCov(r, 'sdPosition', ., 'sdInterval', ., 'sdShift', .)
%  builds the covariance for an ordered tuple of r consecutive
%  differences (intervals) of r + 1 underlying positions:
%
%      Sigma = sdPosition^2 * (D * D')   (tridiagonal: 2 / -1 / -1)
%            + sdInterval^2 * eye(r)     (diagonal)
%            + sdShift^2    * ones(r)    (rank-one ridge)
%
%  Here the intervals are LOG inter-onset intervals (log-IOIs), for one
%  reason: a tempo change t -> a*t multiplies every IOI by a, which in
%  log coordinates is a common ADDITIVE shift of the whole tuple,
%  log(a), along the all-ones diagonal. That makes the three
%  constructor terms three musically distinct tolerances:
%
%    sdPosition   Uncertainty on the underlying positions whose
%                 consecutive differences are the tuple's intervals.
%                 Shared endpoints propagate it to the tridiagonal
%                 sdPosition^2 * (D * D'): displacing one interior
%                 position lengthens one interval and shortens its
%                 neighbour by the same amount. On log-IOIs this
%                 models onset-level timing jitter that scales with
%                 the local inter-onset interval (Weber-like motor
%                 noise); the shared-endpoint reading is exact only
%                 when the adjacent intervals are equal.
%    sdInterval   Independent uncertainty on each interval itself
%                 (central-timekeeper variance in the Wing &
%                 Kristofferson 1973 reading; on log-IOIs,
%                 proportional per-interval noise).
%    sdShift      Graded tolerance for a common shift of the whole
%                 tuple -- on log-IOIs, a TEMPO change. One sd is a
%                 tempo factor of exp(sdShift). As sdShift grows the
%                 kernel's precision tends to the relative-mode
%                 projector, so isRel = true is the exact
%                 (infinite-sdShift) limit: graded tempo TOLERANCE
%                 tends to exact tempo INVARIANCE.
%
%  The demo searches a monophonic onset stream for a long-short-short
%  motif. The stream contains variations of the motif at different
%  tempos, some with small onset-timing perturbations as well, plus two
%  foils. Each candidate rhythm occupies a 3-interval cell; the stream
%  is scanned by its overlapping log-IOI trigrams, each an ordered
%  K = 3 atom multiset read at r = 3 (the matrix covariance requires
%  r == K), with an onset time as a second attribute that only places
%  the sliding window ('windowAttr', 'dropWindowAttr' = true); each
%  trigram is timed at the onset that completes its first interval,
%  the stamp the difference/bind pipeline gives it. The trigram
%  attribute must be ORDERED:
%  one foil is the motif reversed, which has the same interval multiset
%  as the motif and is separated from it only by value order. The
%  trigrams are built with the toolbox's cross-event preprocessing --
%  differenceEvents (onsets to IOIs), then bindEvents (overlapping
%  windows of three consecutive log-IOIs per event). The rel kernel
%  reads each trigram relative to a common shift; a second bindEvents
%  call with 'relOuter' = true produces that reading of the same
%  trigrams.
%
%  Four sections:
%
%    1. Material     The motif, the candidate cells, and the stream.
%    2. Constructor  The three covariance terms, printed, and the
%                    price each pure kernel puts on three canonical
%                    perturbations of the motif.
%    3. The search   windowedSimilarity sweeps over every trigram
%                    under six kernels; the candidate table contrasts
%                    positional (timing) tolerance with tempo
%                    tolerance, and both with exact tempo invariance.
%    4. The limit    sdShift -> infinity converges to isRel = true.
%
%  Similarities throughout are 'normalize' = 'oneSidedDenom', which is
%  1 on a self-match; for these single-trigram comparisons with a
%  shared covariance it equals exp(-delta' * inv(Sigma) * delta / 4),
%  where delta is the difference between the two trigrams' points.
%
%  Two figures are drawn: the onset stream with the query and the six
%  similarity profiles aligned beneath it, and the sdShift sweep
%  converging to the isRel limit.

clear; close all;

% Keep the dispatcher's per-call announcements out of the printed
% tables (showHints gates only those; the one-time truncation notice
% is not gated). The controls are covered in demo_dispatchAndKernelControls.
prevDefaults = mptDefaults('showHints', false);

%% ===== 1. Material =====

% The motif: long-short-short, three IOIs (four onsets), in seconds at
% the reference tempo. All densities are built on natural-log IOIs, so
% a tempo factor a appears as a common shift of log(a).
dMotif = [0.50, 0.25, 0.25];
xMotif = log(dMotif)';   % column: one event, K = 3 values

% Candidate cells. The jittered cells displace the onset shared by the
% two short intervals by +25 ms (at the reference tempo), lengthening
% one short interval and shortening the other -- the signature of
% onset-level jitter. In the jittered-and-faster cell the displacement
% scales with the tempo, consistent with the proportional (log-space)
% jitter model.
cellNames = {'exact', '20% faster', 'double speed', 'jittered', ...
             'jit + faster', 'reversed', 'isochronous'};
cellIois = {dMotif, ...
            dMotif / 1.2, ...
            dMotif / 2.0, ...
            [0.50, 0.275, 0.225], ...
            [0.50, 0.275, 0.225] / 1.2, ...
            [0.25, 0.25, 0.50], ...       % foil: same multiset
            [1, 1, 1] / 3};               % foil: same total duration
nCells = numel(cellIois);

% The stream: the seven cells in order, separated by 1 s of silence.
% Because the gaps exceed every within-cell interval, any trigram that
% straddles a cell boundary reads that 1 s gap as one of its intervals
% -- realistic near-miss material for the search.
gap = 1.0;
onsets = [];
t = 0.0;
for c = 1:nCells
    cellOnsets = t + [0, cumsum(cellIois{c})];
    onsets = [onsets, cellOnsets]; %#ok<AGROW>
    t = cellOnsets(end) + gap;
end

% The input to differenceEvents is a two-attribute pre-MAET both of
% which are the onset times. differenceEvents is told to difference
% the first attribute once and leave the second untouched (the [1, 0]
% orders): the first becomes the inter-onset intervals (each interval
% timed at the onset that completes it), while the second stays as the
% raw onset times. bindEvents then lays a sliding window of three
% consecutive log-IOIs across the event axis (the [3, 1] orders bind
% the interval attribute in threes and keep the time attribute as a
% singleton), one bound event per window, each trigram carrying the
% time of its first interval. The rel kernel needs its trigrams read
% relative to a common shift; the second bind, with 'relOuter' = true,
% produces that reading of the same trigrams.
% The three steps chain as pre-MAETs, each taking the last whole:
% difference the onsets into IOIs, take the log of the interval
% attribute (transformAttributes, which also refuses a zero IOI with its
% remedies rather than yielding -Inf), then bind.
pmDiff = transformAttributes( ...
    differenceEvents({onsets, onsets}, [], [1, 0]), {'log', []});
[pBound, wBound, spBound] = unpackPreMaet(bindEvents(pmDiff, [3, 1]));
pmBoundRel  = bindEvents(pmDiff, [3, 1], 'relOuter', true);
spBoundRel  = pmBoundRel.specs;
% Two quantities read off the bound attributes feed the search below:
nTri = size(pBound{1}, 2);   % number of trigrams (windows to place)
triTimes = pBound{2};        % window-placing times (the sweep centres);
                             % trigram i is timed at onsets(i + 1)

% Presentation only -- no effect on the search, which is blind to cell
% boundaries. cellStarts is the trigram index at which each cell begins,
% used to pick each cell's own trigram for the results table, the figure
% markers, and the near-miss annotation.
cellStarts = 4 * (0:nCells - 1) + 1;

fprintf('\n=== 1. Material ===\n\n');
fprintf('  Motif IOIs (s): [%.2f %.2f %.2f]  (long-short-short)\n', ...
    dMotif);
fprintf(['  Stream: %d cells x 4 onsets, 1 s gaps -> %d onsets, ' ...
    '%d overlapping log-IOI trigrams.\n'], nCells, numel(onsets), nTri);
fprintf('  Each trigram is one event: an ordered K = 3 atom multiset\n');
fprintf('  read at r = 3, timed at the onset that completes its first\n');
fprintf('  interval (the window-placing attribute).\n');

%% ===== 2. intervalKernelCov: shaping the kernel covariance =====

% intervalKernelCov builds the covariance of the trigram attribute's
% Gaussian kernel -- the object that sets how much of each kind of
% departure from the motif the search treats as small. The covariance
% is a sum of three independently scaled terms, one per source of
% uncertainty being smoothed over:
%   sdPosition -- jitter in the underlying onset times. Neighbouring
%     intervals share an onset, so this uncertainty couples them: it
%     enters as a tridiagonal term (2 sd^2 on the diagonal, -sd^2
%     between neighbours).
%   sdInterval -- noise on each interval on its own, uncorrelated
%     across intervals: a diagonal term.
%   sdShift -- a common shift of all three intervals at once, i.e. a
%     tempo change in these log-IOI coordinates: a rank-one ridge
%     (sdShift^2 added to every entry). The larger this ridge, the
%     freer the common shift becomes, and in the limit the kernel
%     approaches the relative-mode reading (isRel = true), which
%     quotients the shift out exactly. Section 4 traces this
%     convergence.
% The three cases below turn on one term at a time so each contribution
% to the covariance is visible on its own.
fprintf('\n=== 2. intervalKernelCov: the kernel covariance ===\n\n');
fprintf(['  intervalKernelCov builds the covariance of the trigram ' ...
    'attribute''s\n']);
fprintf(['  Gaussian kernel from three terms, one per source of ' ...
    'uncertainty\n']);
fprintf(['  the search smooths over: onset-time jitter (sdPosition), ' ...
    'per-\n']);
fprintf(['  interval noise (sdInterval), and a common tempo shift ' ...
    '(sdShift).\n']);
fprintf('  Each case below turns on one term:\n\n');

sPos = intervalKernelCov(3, 'sdPosition', 0.05);
sInt = intervalKernelCov(3, 'sdInterval', 0.05 * sqrt(2));
sRdg = intervalKernelCov(3, 'sdPosition', 0.05, 'sdShift', 0.25);

fprintf(['  sdPosition = 0.05 alone -- onset-time jitter, coupled ' ...
    'across\n']);
fprintf(['  shared onsets (tridiagonal, 2 sd^2 diagonal, -sd^2 ' ...
    'off):\n']);
disp(sPos);
fprintf(['  sdInterval = 0.05*sqrt(2) alone -- independent ' ...
    'per-interval\n']);
fprintf(['  noise (diagonal; chosen to match the tridiagonal''s ' ...
    'per-interval\n']);
fprintf('  marginal variance of 0.005):\n');
disp(sInt);
fprintf(['  sdPosition = 0.05 with sdShift = 0.25 -- onset jitter ' ...
    'plus a\n']);
fprintf('  common-shift ridge (rank-one, added to every entry):\n');
disp(sRdg);

% Price table: perturb the motif along three canonical directions and
% read the similarity of the perturbed trigram to the original under
% each kernel. The directions, in log-IOI units (eps = 0.10):
%   displaced onset:  (0, +eps, -eps)  one interior onset displaced
%                     (the shared endpoint of the two short intervals)
%   one interval:     (0, +eps,  0)    one interval stretched alone
%   tempo shift:      (+eps, +eps, +eps)  every interval scaled by
%                     exp(eps), i.e. a 10.5% tempo change
epsPert = 0.10;
pertNames = {'displaced onset (0,+e,-e)', ...
             'one interval    (0,+e, 0)', ...
             'tempo shift     (e, e, e)'};
perts = {[0; epsPert; -epsPert], [0; epsPert; 0], ...
         [epsPert; epsPert; epsPert]};
pureNames = {'position', 'interval', 'pos+shift'};
pureKernels = {sPos, sInt, sRdg};
w3 = ones(3, 1);

fprintf('\n  Similarity of motif + perturbation to motif (eps = %g):\n\n', ...
    epsPert);
fprintf('  %-28s%7s%12s%12s%12s\n', 'perturbation', '|d|^2', ...
    pureNames{:});
fprintf('  %s\n', repmat('-', 1, 35 + 12 * 3));
for p = 1:numel(perts)
    row = zeros(1, 3);
    for k = 1:3
        row(k) = cosSimExpTens(xMotif, w3, xMotif + perts{p}, w3, ...
            pureKernels{k}, 3, false, false, 0, false, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
    fprintf('  %-28s%7.2f%12.3f%12.3f%12.3f\n', pertNames{p}, ...
        sum(perts{p}.^2), row);
end

fprintf('\n');
fprintf('  Reading the columns:\n');
fprintf('  - position: the displaced onset is CHEAPER than the single\n');
fprintf('    stretched interval despite having twice its squared norm\n');
fprintf('    -- anticorrelated perturbation of adjacent intervals is\n');
fprintf('    exactly what shared-endpoint noise generates, and the\n');
fprintf('    -sd^2 off-diagonals price it accordingly. The tempo shift\n');
fprintf('    is all but forbidden: the sum of the r intervals equals\n');
fprintf('    the difference of the two endpoint positions, so its\n');
fprintf('    variance under position noise is 2 sd^2 regardless of r\n');
fprintf('    -- a common drift of all intervals is highly atypical of\n');
fprintf('    position noise.\n');
fprintf('  - interval: pricing is by Euclidean norm alone (the two\n');
fprintf('    marginals are matched to the position column), so the\n');
fprintf('    ordering of the first two rows reverses.\n');
fprintf('  - pos+shift: the ridge makes the tempo shift the cheapest\n');
fprintf('    direction while leaving the within-shape prices\n');
fprintf('    essentially unchanged.\n');

%% ===== 3. The search: positional sigma vs tempo sigma =====

fprintf('\n=== 3. Searching the stream for the motif ===\n\n');

% Six kernels. sd values are in natural-log units: sdPosition = 0.10
% tolerates onset jitter of roughly 10% of the local inter-onset
% interval; sdShift = 0.25 makes one sd a tempo factor of
% exp(0.25) ~ 1.28 (or its reciprocal). The rel entry is exact tempo
% invariance; its scalar sigma = 0.10*sqrt(2) matches the timing
% kernel's per-interval marginal (2 * sdPosition^2).
kernelNames = {'strict', 'timing', 'tempo', 'timing+tempo', ...
               'large-shift', 'rel'};
kernelSigmas = {intervalKernelCov(3, 'sdPosition', 0.02), ...
                intervalKernelCov(3, 'sdPosition', 0.10), ...
                intervalKernelCov(3, 'sdPosition', 0.02, ...
                                  'sdShift', 0.25), ...
                intervalKernelCov(3, 'sdPosition', 0.10, ...
                                  'sdShift', 0.25), ...
                intervalKernelCov(3, 'sdInterval', 0.10 * sqrt(2), ...
                                  'sdShift', 100), ...
                0.10 * sqrt(2)};
kernelSpecs = {spBound{1}, spBound{1}, spBound{1}, spBound{1}, ...
               spBound{1}, spBoundRel{1}};
fprintf('  strict       : intervalKernelCov(3, ''sdPosition'', 0.02)\n');
fprintf('  timing       : intervalKernelCov(3, ''sdPosition'', 0.10)\n');
fprintf(['  tempo        : intervalKernelCov(3, ''sdPosition'', 0.02, ' ...
    '''sdShift'', 0.25)\n']);
fprintf(['  timing+tempo : intervalKernelCov(3, ''sdPosition'', 0.10, ' ...
    '''sdShift'', 0.25)\n']);
fprintf(['  large-shift  : intervalKernelCov(3, ''sdInterval'', ' ...
    '0.10*sqrt(2), ''sdShift'', 100)\n']);
fprintf(['  rel          : relOuter=true bind spec, sigma = ' ...
    '0.10*sqrt(2)  (exact tempo invariance)\n']);
fprintf('  The large-shift kernel''s within-shape term is matched to\n');
fprintf('  the rel kernel''s sigma, so its enormous ridge should\n');
fprintf('  reproduce the rel column almost exactly (Section 4 gives\n');
fprintf('  the limit argument).\n');

% One windowedSimilarity sweep per kernel. The rect window (full width
% 0.1 s, narrower than the smallest trigram spacing of 0.125 s)
% restricts each comparison to the single trigram at its centre; the
% time attribute only places the window and is dropped from the
% comparison, so each profile value is the plain similarity of that
% trigram to the query under the kernel. The search itself uses no
% knowledge of where the cells sit: every event of the stream starts a
% candidate trigram and receives a window, boundary-straddling
% trigrams included. Window centres are anchored
% to the trigram times rather than laid on a uniform grid, so
% their spacing follows the stream's own inter-onset intervals --
% denser where the music is faster, widest across the silences. A
% uniform grid (available via 'start'/'stop'/'step') would add nothing
% at this window width: a window containing one trigram returns that
% trigram's similarity wherever within its span the window is placed,
% and a window containing none returns zero, its context density
% being empty.
pContext = pBound;         % {trigrams, times} as bound
wContext = {ones(3, nTri), ones(1, nTri)};
pQuery = {xMotif, 0};
wQuery = {ones(3, 1), 1};

showPreMaet(pContext, wContext, {kernelSpecs{1}, spBound{2}}, ...
    'names', {'trigram', 'time'}, 'sigma', {kernelSigmas{1}, 0.25}, ...
    'isPer', [false false], 'maxEvents', 4);
showPreMaet(pQuery, wQuery, {kernelSpecs{1}, spBound{2}}, ...
    'names', {'trigram', 'time'}, 'sigma', {kernelSigmas{1}, 0.25}, ...
    'isPer', [false false]);
fprintf('\n');

profiles = zeros(numel(kernelNames), nTri);
for k = 1:numel(kernelNames)
    profiles(k, :) = windowedSimilarity(pContext, wContext, ...
        pQuery, wQuery, {kernelSigmas{k}, 0.25}, [3, 1], ...
        [false, false], [false, false], [0, 0], triTimes, ...
        'specs', {kernelSpecs{k}, spBound{2}}, 'windowAttr', 2, ...
        'dropWindowAttr', true, 'contextWindow', {'rect', 0.1}, ...
        'normalize', 'oneSidedDenom', 'verbose', false);
end

fprintf('\n  Profile at each cell''s own trigram:\n\n');
fprintf('  %-14s%14s%14s%14s%14s%14s%14s\n', 'candidate', kernelNames{:});
fprintf('  %s\n', repmat('-', 1, 14 + 14 * 6));
for c = 1:nCells
    fprintf('  %-14s%14.3f%14.3f%14.3f%14.3f%14.3f%14.3f\n', ...
        cellNames{c}, profiles(:, cellStarts(c)));
end

fprintf('\n');
fprintf('  Reading the rows:\n');
fprintf('  - 20%% faster / double speed: pure tempo changes. Positional\n');
fprintf('    sigma alone barely admits them at any tolerable width\n');
fprintf('    (''timing'' gives 0.016 at sdPosition = 0.10); sdShift\n');
fprintf('    admits the moderate change and GRADES the large one\n');
fprintf('    (''tempo'' gives 0.876 and 0.147); rel admits both\n');
fprintf('    exactly.\n');
fprintf('  - jittered: a same-tempo timing perturbation. Tempo sigma\n');
fprintf('    alone does not help (''tempo'' gives 0.011); positional\n');
fprintf('    sigma does (''timing'' gives 0.832). The two tolerances\n');
fprintf('    are separate currencies: in log-IOI space a tempo change\n');
fprintf('    moves the trigram''s point ALONG the all-ones diagonal,\n');
fprintf('    timing jitter moves it off that line, and the kernel\n');
fprintf('    prices the two components independently.\n');
fprintf('  - jit + faster: needs both currencies at once -- only the\n');
fprintf('    kernels carrying both, ''timing+tempo'' (0.742),\n');
fprintf('    ''large-shift'', and ''rel'' (0.777 each), admit it.\n');
fprintf('    Under ''rel'' the two jittered rows are identical: the\n');
fprintf('    jittered-and-faster cell is the jittered cell under a\n');
fprintf('    pure tempo change (its displacement scales with the\n');
fprintf('    tempo), and rel quotients tempo out.\n');
fprintf('  - reversed: same interval multiset as the motif; the\n');
fprintf('    ordered outer read (symOuter = false, the bind default)\n');
fprintf('    keeps it at zero under every kernel.\n');
fprintf('  - isochronous: a genuinely different shape; near zero\n');
fprintf('    throughout.\n');
fprintf('  - large-shift vs rel: the two columns agree at this\n');
fprintf('    precision. Graded tolerance with a sufficiently large\n');
fprintf('    ridge is numerically indistinguishable from the exact\n');
fprintf('    quotient -- the limit Section 4 approaches from below,\n');
fprintf('    effectively reached.\n\n');
fprintf('  Max |large-shift - rel| across all %d trigram positions: %.1e\n\n', ...
    nTri, max(abs(profiles(5, :) - profiles(6, :))));

% The full profile also sweeps the boundary-straddling trigrams. One
% is instructive: the trigram reading (last interval of the reversed
% cell, the 1 s gap, first isochronous interval) = (0.50, 1.0, 0.33) s
% -- the silence itself parses as the 'long' of a long-short-short
% figure with ratio 3:1:1, close in shape to the motif's 2:1:1 but at
% a remote tempo. That trigram is the one immediately before the
% isochronous cell's own.
iStraddle = cellStarts(end) - 1;
fprintf(['  A boundary near-miss: the trigram at t = %.2f s reads ' ...
    'the rest\n'], triTimes(iStraddle));
fprintf('  after the reversed cell as a ''long'', giving a\n');
fprintf('  long-short-short of ratio 3:1:1 at a remote tempo. Graded\n');
fprintf('  tempo tolerance suppresses what exact invariance admits:\n');
for k = [3, 4, 5, 6]
    fprintf('    %-13s: %.3f\n', kernelNames{k}, profiles(k, iStraddle));
end

% --- Main figure: the stream, the query, and the six profiles ------
% Top tile: the onset stream as an event raster, with each cell's span
% shaded and named, and the query drawn on a second y-level at
% t = 0..1 s -- the same time scale, sitting directly above the exact
% copy for visual comparison. Below: one tile per kernel, sharing the
% time axis, with the cell spans repeated so each peak reads off
% against its cell. Each profile is drawn as end-aligned stair steps:
% a trigram's tread spans its first inter-onset interval and its value
% sits at the right edge (the trigram's stamp), with a dot marking
% each stamp. Tile titles carry each kernel's constructor parameters.
kernelParamLabels = {'sdPosition = 0.02', ...
                     'sdPosition = 0.10', ...
                     'sdPosition = 0.02, sdShift = 0.25', ...
                     'sdPosition = 0.10, sdShift = 0.25', ...
                     'sdInterval = 0.10*sqrt(2), sdShift = 100', ...
                     'relOuter = true, sigma = 0.10*sqrt(2)'};
cellSpans = zeros(nCells, 2);
for c = 1:nCells
    cellSpans(c, :) = [onsets(4 * (c - 1) + 1), onsets(4 * (c - 1) + 4)];
end
queryOnsets = [0, cumsum(dMotif)];
xLim = [-0.6, onsets(end) + 0.6];
blue = [0, 0.4470, 0.7410];
red = [0.84, 0.15, 0.16];

fig1 = figure('Units', 'pixels', 'Position', [80, 80, 1100, 980], ...
    'Color', 'w');
tl = tiledlayout(fig1, 15, 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
allAx = gobjects(1, 7);

% Context + query tile (spans 3 rows).
ax = nexttile(tl, [3, 1]);
allAx(1) = ax;
hold(ax, 'on');
for c = 1:nCells
    patch(ax, cellSpans(c, [1, 2, 2, 1]), [0, 0, 3.3, 3.3], ...
        [0.55, 0.55, 0.55], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end
nOn = numel(onsets);
xs = [onsets; onsets; nan(1, nOn)];
ys = [zeros(1, nOn); ones(1, nOn); nan(1, nOn)];
plot(ax, xs(:), ys(:), '-', 'Color', 'k', 'LineWidth', 1.2);
nQ = numel(queryOnsets);
xq = [queryOnsets; queryOnsets; nan(1, nQ)];
yq = [1.55 * ones(1, nQ); 2.45 * ones(1, nQ); nan(1, nQ)];
plot(ax, xq(:), yq(:), '-', 'Color', red, 'LineWidth', 1.8);
text(ax, queryOnsets(end) + 0.15, 2.0, 'query', 'Color', red, ...
    'FontSize', 9, 'VerticalAlignment', 'middle');
for c = 1:nCells
    text(ax, mean(cellSpans(c, :)), 1.12, cellNames{c}, ...
        'FontSize', 8, 'Rotation', 30, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
end
ylim(ax, [0, 3.3]);
set(ax, 'YTick', [], 'XTickLabel', []);
ylabel(ax, 'events', 'FontSize', 9);

% Profile tiles (2 rows each). Left-aligned stair treads: tread i
% starts at triTimes(i) and holds until triTimes(i + 1); the final
% tread runs to the end of the last trigram.
for k = 1:numel(kernelNames)
    ax = nexttile(tl, [2, 1]);
    allAx(k + 1) = ax;
    hold(ax, 'on');
    for c = 1:nCells
        patch(ax, cellSpans(c, [1, 2, 2, 1]), ...
            [-0.07, -0.07, 1.30, 1.30], [0.55, 0.55, 0.55], ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end
    % End-aligned stair steps: the tread for a trigram spans its first
    % inter-onset interval -- from the trigram's opening onset to the
    % onset that stamps it (onsets(i) to onsets(i+1)) -- so the tread
    % width shows that interval and the value sits at its right edge.
    % For the boundary trigram this first interval is the 1 s gap, so
    % the tread widens to cover it. Small dots mark the stamps; the
    % leading edge is carried back to the opening onset. The last value
    % is repeated so its tread closes at the final stamp.
    stairs(ax, [onsets(1), triTimes], [profiles(k, :), profiles(k, end)], ...
        '-', 'Color', blue, 'LineWidth', 1.0);
    plot(ax, triTimes, profiles(k, :), '.', 'Color', blue, ...
        'MarkerSize', 8);
    text(ax, xLim(1) + 0.01 * diff(xLim), 1.26, ...
        sprintf('%s (%s)', kernelNames{k}, kernelParamLabels{k}), ...
        'FontSize', 9, 'FontWeight', 'bold', ...
        'VerticalAlignment', 'top');
    ylim(ax, [-0.07, 1.30]);
    set(ax, 'YTick', [0, 0.5, 1]);
    ylabel(ax, 'similarity', 'FontSize', 9);
    if k < numel(kernelNames)
        set(ax, 'XTickLabel', []);
    end
end
% Annotate the boundary near-miss wherever tempo invariance admits it:
% the wide gap tread reads as a 'long', so both the large-shift and rel
% tiles score it. Text sits up and to the right of the point so the
% pointer stays short.
annKernels = {'large-shift', 'rel'};
for a = 1:numel(annKernels)
    kIdx = find(strcmp(kernelNames, annKernels{a}));
    axA = allAx(kIdx + 1);
    xP = triTimes(iStraddle);
    yP = profiles(kIdx, iStraddle);
    plot(axA, [xP - 0.05, xP], [0.62, yP + 0.03], '-', ...
        'Color', 'k', 'LineWidth', 0.5);
    text(axA, xP - 0.05, 0.66, 'gap parses as ''long''', ...
        'FontSize', 8, 'HorizontalAlignment', 'left');
end
xlabel(allAx(end), 'time (s)');
linkaxes(allAx, 'x');
xlim(allAx(1), xLim);

%% ===== 4. From tolerance to invariance: the isRel limit =====

fprintf('\n=== 4. sdShift -> infinity is isRel = true ===\n\n');
fprintf('  As sdShift grows, the kernel''s precision tends to the\n');
fprintf('  relative-mode projector: the shift direction becomes free\n');
fprintf('  while the within-shape metric is left behind. With the\n');
fprintf('  within-shape term supplied by sdInterval = 0.08, the limit\n');
fprintf('  is EXACTLY isRel = true at sigma = 0.08, because rel\n');
fprintf('  mode''s isotropic within-shape kernel is the limit of the\n');
fprintf('  diagonal (sdInterval) family. (The sdPosition family also\n');
fprintf('  has a shift-invariant limit, but its within-shape metric\n');
fprintf('  is the tridiagonal restricted to the zero-sum subspace,\n');
fprintf('  which is not isotropic, so no scalar rel sigma reproduces\n');
fprintf('  it.)\n\n');

sdInt = 0.08;
targetNames = {'double speed', 'jit + faster'};
targetVals = {log(dMotif / 2.0)', log([0.50, 0.275, 0.225] / 1.2)'};
shifts = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0];

fprintf('  %-12s%15s%15s\n', 'sdShift', targetNames{:});
fprintf('  %s\n', repmat('-', 1, 12 + 15 * 2));
for ss = shifts
    s = intervalKernelCov(3, 'sdInterval', sdInt, 'sdShift', ss);
    row = zeros(1, 2);
    for j = 1:2
        row(j) = cosSimExpTens(xMotif, w3, targetVals{j}, w3, s, 3, ...
            false, false, 0, false, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
    fprintf('  %-12g%15.4f%15.4f\n', ss, row);
end
row = zeros(1, 2);
for j = 1:2
    row(j) = cosSimExpTens(xMotif, w3, targetVals{j}, w3, sdInt, 3, ...
        true, false, 0, false, ...
        'normalize', 'oneSidedDenom', 'verbose', false);
end
fprintf('  %-12s%15.4f%15.4f\n', 'isRel = true', row);

% --- Limit figure: similarity vs sdShift, with the rel asymptotes ---
ssDense = logspace(log10(0.05), log10(20.0), 60);
cols = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980];
fig2 = figure('Units', 'pixels', 'Position', [120, 120, 620, 420], ...
    'Color', 'w');
ax2 = axes(fig2);
hold(ax2, 'on');
hCurves = gobjects(1, 2);
for j = 1:2
    curve = zeros(size(ssDense));
    for i = 1:numel(ssDense)
        s = intervalKernelCov(3, 'sdInterval', sdInt, ...
            'sdShift', ssDense(i));
        curve(i) = cosSimExpTens(xMotif, w3, targetVals{j}, w3, s, ...
            3, false, false, 0, false, ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
    hCurves(j) = plot(ax2, ssDense, curve, '-', ...
        'Color', cols(j, :), 'LineWidth', 1.2);
    yline(ax2, row(j), '--', 'Color', cols(j, :), 'LineWidth', 0.9);
    text(ax2, 0.055, row(j) + 0.02, ...
        sprintf('isRel = true: %.3f', row(j)), ...
        'Color', cols(j, :), 'FontSize', 8);
end
set(ax2, 'XScale', 'log');
xlim(ax2, [0.05, 20]);
ylim(ax2, [-0.03, 1.08]);
xlabel(ax2, 'sdShift (log scale)');
ylabel(ax2, 'similarity');
title(ax2, sprintf(['Tempo tolerance -> tempo invariance ' ...
    '(sdInterval = %g)'], sdInt), 'FontSize', 10, ...
    'FontWeight', 'normal');
legend(hCurves, targetNames, 'Location', 'east', 'FontSize', 9);

fprintf('\n');
fprintf('  The double-speed copy converges to 1 (a pure shift is\n');
fprintf('  fully absorbed); the jittered-and-faster copy converges to\n');
fprintf('  the rel value set by its within-shape (jitter) component\n');
fprintf('  alone. Tempo tolerance varies continuously with sdShift;\n');
fprintf('  tempo invariance is its limit.\n');

mptDefaults(prevDefaults);
