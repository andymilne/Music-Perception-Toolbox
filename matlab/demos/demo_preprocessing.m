%% demo_preprocessing.m
% Pre-MAET preprocessing operations and their compositions.
%
% Demonstrates the pre-MAET preprocessing operations, applied to a
% fragment of J. S. Bach, BWV 347 ("Ich dank dir, lieber Herre"): the
% soprano over quarter-notes t = 1 to 7, which is bar 1 entire followed
% by cadence 1's three-chord approach (antepenult i, penult V, tonic I
% at t = 5, 6, 7), so the fragment ends on its cadential goal. Two
% attributes are kept: the soprano pitch (attribute 1, treated as
% periodic mod 12 so it lives on the pitch-class circle) and the event
% time in quarter-notes (attribute 2, non-periodic). The pre-MAET
% carries each attribute's kernel geometry (sigma, periodicity, and
% period) in its specs, so every operation below takes it whole and
% returns it whole, and the tensor functions read the geometry from it.
% Each subsequent section illustrates one operation or one composition;
% the operations leave the source pre-MAET untouched.
%
% The events carry metrical weights rather than uniform ones, so that
% each operation's weight rule is visible in its output rather than
% described: the chorale is in 4/4 with a one-quarter pickup, so t = 1
% and t = 5 fall on the downbeat (weight 1), t = 3 and t = 7 on the
% third beat (0.75), and t = 2, 4, 6 on the weak beats (0.5).
%
% Every operation's result is displayed with showPreMaet, which prints a
% pre-MAET in the layout of the article's tables: brace-delimited cells
% where the attribute is unordered, parentheses where it is ordered,
% brackets within brackets where it is nested, and weights as
% parenthesized superscripts.
%
%   Operations
%       differenceEvents    (D): event differencing, with per-attribute
%                                difference orders.
%       bindEvents          (B): event binding, with per-attribute bind
%                                orders (n-grams of consecutive events).
%       translateAttributes (T): attribute translation.
%       weightEvents        (W): event weighting, a per-event window (one
%                                factor per input attribute, from a
%                                peak-normalized family of fixed
%                                variance).
%       selectPreMaet       (S): a selection of attributes and events.
%       bindAttributes,          one attribute from several, and several
%       separateAttributes:      from one.
%       transformAttributes (F): attribute rescaling, by per-attribute
%                                elementwise maps (log, scale conversion,
%                                user function), with an optional sign
%                                attribute.
%
%   Compositions
%       D o B == B o D      (event differencing and event binding
%                            commute, in values, weights, and specs).
%       D o T == D          (differencing absorbs attribute translation;
%                            T o D adds mu to every difference).
%       T o W centre shift  (W with centre c after T(mu) equals W with
%                            centre c - mu before T; T leaves weights
%                            unchanged, and W leaves values unchanged).
%       D o F vs F o D      (log then difference gives log ratios;
%                            difference then log(x + 1) with a sign
%                            attribute gives signed compressed
%                            magnitudes).
%
% Spectral enrichment (addSpectra), the sixth preprocessing operation of
% the article, is demonstrated in jmm/demo_jmm_2_3_spectral.m.
%
%   Where the operations are taken further
%       sweptSimilarity,     translation and event weighting swept along
%       sweptEntropy:        a piece, a similarity or an entropy at each
%                            sweep value (demo_sweptSimilarity;
%                            jmm/demo_jmm_1_1_entropy).
%       sweepSimMaet:        a translation sweep on built densities, in
%                            one pass.
%       nTupleEntropy:       the D o B pipeline of Section 6, packaged
%                            (demo_rhythmTensors, demo_sigmaSpace).
%       demo_tempoInvariance, demo_repetitionHandling:
%                            D, B, and F composed for tempo and
%                            interval-scale invariance.
%       demo_scoreWorkflow, demo_scoreCategoricals:
%                            a pre-MAET built from a score, with
%                            selectPreMaet and separateAttributes.
%       demo_preMaetIo:      showing, writing, and reading a pre-MAET.
%
% See also SHOWPREMAET, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES,
% WEIGHTEVENTS, SELECTPREMAET, BINDATTRIBUTES, SEPARATEATTRIBUTES,
% TRANSFORMATTRIBUTES, SWEPTSIMILARITY, SWEPTENTROPY, SWEEPSIMMAET.
%
% The Python mirror is demo_preprocessing.py.

clear; clc;

% The toolbox's one-time informational hints (which route a call took,
% and the like) are switched off for a tidy printout, and restored at
% the end.
prevDefaults = mptDefaults('showHints', false);

%% ===================================================================
%  1. BWV 347 input (soprano + time, metrically weighted)
%  ===================================================================

fprintf('=== 1. Inputs (BWV 347, soprano, t = 1..7) ===\n');

% Two attributes, both K_a = 1, seven events.
pAttr = { [69 69 69 71 67 66 64], ...      % a_1: soprano
          [ 1  2  3  4  5  6  7] };        % a_2: time (quarter-notes)

% Metrical weight: downbeat 1, third beat 0.75, weak beats 0.5. The same
% weighting is carried on both attributes, since a metrically weak event
% is weak in every attribute it carries.
metre = [1 0.5 0.75 0.5 1 0.5 0.75];
w = {metre, metre};

% The kernel geometry of each attribute: pitch periodic at the octave
% (12 semitones) with sigma = 0.5 semitones, and time non-periodic with
% sigma = 0.25 quarter-notes. Both attributes are absolute.
specs = flatSpecs(pAttr, 'name', {'pitch', 'time'}, 'sigma', [0.5 0.25], ...
                  'isPer', [true false], 'period', [12 0]);

% The three parts travel together as one pre-MAET, which every operation
% below takes whole and returns whole.
pm = packPreMaet(pAttr, w, specs);

showPreMaet(pm);
fprintf('\n');

%% ===================================================================
%  2. differenceEvents (D): turn pitches into intervals
%  ===================================================================

fprintf('=== 2. differenceEvents (D) ===\n');

% Take the first difference of pitch and leave time alone. The first
% event is dropped (leading-drop alignment): N' = N - max(k) = 6.
% Weights propagate as the rolling product w'(n) = prod_{j=0..k} w(n-j),
% the probability that the k+1 contributing events are jointly
% perceived, so a difference is only as strong as its weaker endpoint
% allows: the superscripts below are the products of consecutive metre
% weights on the differenced attribute, and the surviving metre weights
% on the undifferenced one. The differenced attribute's sigma grows by
% sqrt(2), since a difference of two uncertain values is less certain
% than either; differenceEvents says so as it runs. Differencing is how
% interval (transposition-invariant) and inter-onset (time-shift-
% invariant) content is obtained: demo_overview, section 2c, uses it to
% find a motif at any transposition, and demo_tempoInvariance and
% demo_repetitionHandling build on it.
diffOrders = [1 0];
pmD = differenceEvents(pm, diffOrders);

fprintf('  diffOrders = [%d %d]   (pitch differenced, time left alone)\n', ...
    diffOrders(1), diffOrders(2));
showPreMaet(pmD);
fprintf('\n');

%% ===================================================================
%  3. bindEvents (B): expand attributes into 2-grams
%  ===================================================================

fprintf('=== 3. bindEvents (B) ===\n');

% Bind 2 consecutive events into 2-grams on both attributes. Each source
% attribute becomes ONE nested attribute: the two bound events form the
% ordered outer level, each event's own value the inner level (inheriting
% the source r/isRel/isExch). A' = A = 2. Trailing-drop alignment gives
% N' = N - max(L) + 1 = 6. B gathers each super-event's constituent
% weights alongside its values rather than combining them, so both of a
% 2-gram's metre weights survive, in order, inside the cell. Binding is
% how n-grams and nested multisets are compared: nTupleEntropy is
% differencing then binding (Section 6), and
% jmm/demo_jmm_1_3_cadence_nesting finds cadences with nested bound
% events.
bindOrders = [2 2];
pmB = bindEvents(pm, bindOrders);
specB = pmB.specs;

fprintf(['  bindOrders = [%d %d]   (A'' = %d: each source attribute ' ...
    '-> one nested attribute)\n'], bindOrders(1), bindOrders(2), ...
    numel(pmB.pAttr));
showPreMaet(pmB);
fprintf('  spec(1): r = [%s], exch = [%s], rel = [%s], tags = [%s]\n', ...
    num2str(specB{1}.r), num2str(double(specB{1}.exch)), ...
    num2str(double(specB{1}.rel)), num2str(specB{1}.tags(:)'));
fprintf('\n');

%% ===================================================================
%  3b. bindEvents again (B o B): deepen a nested attribute to L = 3
%  ===================================================================

fprintf('=== 3b. bindEvents again (B o B): deepen to L = 3 ===\n');

% bindEvents accepts the pre-MAET it produces, so a second bind deepens
% the *already-nested* attribute rather than starting over. The specs
% travel with the pre-MAET, so nothing has to be threaded by hand. The
% existing tag matrix is tiled and a fresh outermost grouping column is
% appended; r/exch/rel each gain one outer level. The hierarchy grows
% note -> 2-event group (first bind) -> 2-group window (second bind).
% Trailing-drop again: N'' = N' - max(L) + 1 = 5.
pmBB = bindEvents(pmB, [2 2]);
specBB = pmBB.specs;

fprintf('  N'''' = %d  (three-level super-events)\n', size(pmBB.pAttr{1}, 2));
showPreMaet(pmBB, 'maxEvents', 4);
fprintf('  spec(1): r = [%s], exch = [%s], rel = [%s]\n', ...
    num2str(specBB{1}.r), num2str(double(specBB{1}.exch)), ...
    num2str(double(specBB{1}.rel)));
fprintf('  (inner tag column tiled; a new outermost column appended.)\n');

% Build the absolute L = 3 nest and confirm a clean self-similarity. The
% kernel geometry travelled through both binds in the specs, so nothing
% further is supplied here; an argument given at the call would override
% the spec, which is what makes a sweep one call per value
% (demo_preMaetIo, section 4).
dBBAbs = buildMaet(pmBB, 'verbose', false);
smAbs = simMaet(dBBAbs, dBBAbs, 'verbose', false);
fprintf('  absolute build: dim = %d, cosine self-match = %.4f\n', ...
    dBBAbs.dim, smAbs);

% Making the outermost level of pitch relative ([rel] = 1) reads the
% whole three-level tuple up to a common shift, so the doubly-bound
% pitch structure is invariant to transposing every note together.
% Absolute pitch is not: the narrow pitch kernel (sigma = 0.5) puts a
% 5-semitone shift out of reach. Replacing one part of a pre-MAET leaves
% the rest in place: here the specs.
specBBOut = specBB;
specBBOut{1}.rel = [0 0 1];                  % outermost unit on pitch
pmBBOut = packPreMaet(pmBB, [], specBBOut);
dBBOut = buildMaet(pmBBOut, 'verbose', false);
pmBBT = translateAttributes(pmBB, {5, 0});   % every pitch +5
pmBBOutT = packPreMaet(pmBBT, [], specBBOut);
dBBOutT = buildMaet(pmBBOutT, 'verbose', false);
dBBAbsT = buildMaet(pmBBT, 'verbose', false);
simOut = simMaet(dBBOut, dBBOutT, 'verbose', false);
simAbs = simMaet(dBBAbs, dBBAbsT, 'verbose', false);
fprintf(['  outer pitch (rel=[0 0 1]): dim = %d, vs +5 transpose = %.4f' ...
    '  (global-transposition invariant)\n'], dBBOut.dim, simOut);
fprintf(['  absolute pitch:            vs +5 transpose = %.4f' ...
    '  (not invariant)\n\n'], simAbs);

%% ===================================================================
%  4. translateAttributes (T): transpose pitch up a perfect fourth
%  ===================================================================

fprintf('=== 4. translateAttributes (T) ===\n');

% Translate pitch (attribute 1) by +5 semitones; leave time alone.
% Offsets are a per-attribute cell: a scalar broadcasts across the
% attribute's values (here K = 1 each). isRel is read from the specs
% (both attributes absolute), so neither translation is a no-op. T moves
% values only: the weights below are the metre weights unchanged.
%
% One call makes one translation. To compare a query with a context at
% each of many translations -- a sliding comparison, the canonical use of
% translation -- use sweptSimilarity (pre-MAETs; demo_sweptSimilarity) or
% sweepSimMaet (densities), which compute every offset in one pass rather
% than building a copy per offset.
muPitch = 5;
mu = {muPitch, 0};
pmT = translateAttributes(pm, mu);

fprintf('  mu (per attribute) = {%g, %g}   (G->C, F#->B, E->A; time untouched)\n', ...
    mu{1}, mu{2});
showPreMaet(pmT);
fprintf('\n');

%% ===================================================================
%  5. weightEvents (W): window the time attribute at the cadence
%  ===================================================================

fprintf('=== 5. weightEvents (W) ===\n');

% Apply a window on the time attribute (input attribute 2) centred at the
% penult event (t = 6) with standard deviation 2 quarter-notes and
% gamma = 0 (pure Gaussian). The factor lands back on the time attribute
% (target attribute 2), the in-place weighting case, and the input is
% kept (dropInputAttr = false). The window multiplies the metre weights
% it finds rather than replacing them, so the time row below carries
% metre times envelope, and the pitch row is untouched.
%
% One call weights the events at one position. Sweeping a window along
% a piece, with an entropy or a similarity at each position, is
% sweptEntropy, or sweptSimilarity with 'align' 'window'
% (demo_sweptSimilarity, sections 5 to 8; jmm/demo_jmm_1_1_entropy).
pmW = weightEvents(pm, 2, 2, 6, 0, 'sd', 2, 'dropInputAttr', false);

fprintf(['  inputAttr = 2 (time); targetAttr = 2; centre = 6; sd = 2; ' ...
    'shape = 0 (Gaussian)\n']);
showPreMaet(pmW, 'decimals', 3);
fprintf('\n');

%% ===================================================================
%  5b. selectPreMaet (S): keep some attributes and some events
%  ===================================================================

fprintf('=== 5b. selectPreMaet (S) ===\n');

% A filter on the pre-MAET itself, as against selecting rows of the
% attribute table it may have been built from, which is MATLAB's own
% job. It reads only the two levels every pre-MAET has -- its attributes
% and its events -- so it knows nothing of where the pre-MAET came from.
% Here the cadence's three chords (events 5 to 7) on the pitch attribute
% alone; the kept items come back in the order given, and each keeps its
% tuple size and flags, so a selection cannot change what an attribute
% means. demo_scoreWorkflow and demo_scoreCategoricals use it on
% pre-MAETs read from a score.
pmS = selectPreMaet(pm, 'attributes', 1, 'events', 5:7);

fprintf('  attributes = 1 (pitch); events = 5:7 (the cadence)\n');
showPreMaet(pmS, 'decimals', 3);
fprintf('\n');

%% ===================================================================
%  5c. bindAttributes and separateAttributes: one attribute from
%      several, and several from one
%  ===================================================================

fprintf('=== 5c. bindAttributes and separateAttributes ===\n');

% Binding across attributes, as bindEvents binds across events. Pitch
% and time describe the same events, so binding them gives one
% attribute whose value at an event is the ordered pair, read whole
% (r = 2, exch = false) rather than as the product of two attributes.
% The two disagree on periodicity, so the bound attribute is given its
% own: non-periodic, with one sigma for both slots.
pmBA = bindAttributes(pm, [1 2], 'name', 'pitchTime', 'r', 2, ...
                      'exch', false, 'sigma', 0.5, 'isPer', false, ...
                      'period', 0);
showPreMaet(pmBA, 'decimals', 3);
fprintf('\n');

% separateAttributes goes the other way, splitting the bound attribute
% into one attribute per slot, each named for the bound attribute and
% its slot.
[pBack, ~, sBack] = unpackPreMaet(separateAttributes(pmBA, 'pitchTime'));
fprintf('  separated into %d attributes: %s\n\n', numel(pBack), ...
        strjoin(cellfun(@(x) x.name, sBack, 'UniformOutput', false), ', '));

%% ===================================================================
%  6. D o B == B o D (n-tuple entropy pipeline commutation)
%  ===================================================================

fprintf('=== 6. D o B == B o D (event differencing and event binding commute) ===\n');

% Both operations take a pre-MAET and return one, so they compose
% directly and the two routes coincide. Differencing pairs values
% position by position across (super-)events and the sliding bind
% window commutes with it, on the ordered, K = 1 domain where a
% difference is defined. The two operations propagate weights by
% different rules --- D takes the rolling product, B gathers --- and the
% composition agrees on the weights as well. This pipeline is the n-tuple
% entropy of Milne and Dean (2016), which nTupleEntropy packages
% (demo_rhythmTensors, demo_sigmaSpace).
%   D then B: difference each attribute (order 1), then bind 2-grams.
pmDB = bindEvents(differenceEvents(pm, [1 1]), [2 2]);
%   B then D: bind 2-grams, then difference each nested attribute
%   position by position.
pmBD = differenceEvents(bindEvents(pm, [2 2]), [1 1]);

showPreMaet(pmDB, 'title', '  D then B:');
showPreMaet(pmBD, 'title', '  B then D:');

valsAgree = true; wtsAgree = true; specsAgree = true;
for a = 1:2
    valsAgree = valsAgree && isequaln(pmDB.pAttr{a}, pmBD.pAttr{a});
    wtsAgree = wtsAgree && isequaln(pmDB.wAttr{a}, pmBD.wAttr{a});
    for f = {'tags', 'r', 'exch', 'rel', 'sigma'}
        specsAgree = specsAgree && ...
            isequal(double(pmDB.specs{a}.(f{1})(:)'), ...
                    double(pmBD.specs{a}.(f{1})(:)'));
    end
end
fprintf('  values agree: %d;  weights agree: %d;  specs agree: %d\n', ...
    valsAgree, wtsAgree, specsAgree);
assert(valsAgree && wtsAgree && specsAgree, ...
    'demo_preprocessing:commute', 'Section 6: the two routes disagree.');
fprintf('\n');

%% ===================================================================
%  7. D o T == D (differencing absorbs absolute translation)
%  ===================================================================

fprintf('=== 7. D o T == D ===\n');

% Difference applied to a transposed copy returns the same intervals as
% differencing the original: translation is wiped out by the difference
% operator (T o D, by contrast, adds mu to every difference).
pmDT = differenceEvents(pmT, [1 0]);

showPreMaet(pmDT, 'title', '  D(T(p)):');
fprintf(['  vs D(p) above: max |difference| = %g  ' ...
    '(zero: translation absorbed)\n\n'], ...
    max(abs(pmDT.pAttr{1} - pmD.pAttr{1})));

%% ===================================================================
%  8. T o W centre-shift rule
%  ===================================================================

fprintf('=== 8. T o W centre shift ===\n');

% Path 1: T(mu) first (transposing pitch by +5), then W centred at the
% original pitch c = 69 (A4).
cPitch = 69;
widthW = 2;
gammaW = 0.3;
pmPath1 = weightEvents(translateAttributes(pm, {muPitch, 0}), ...
    1, 1, cPitch, gammaW, 'sd', widthW, 'dropInputAttr', false);

% Path 2: W centred at c - mu = 64 BEFORE T (T leaves weights untouched).
pmPath2 = weightEvents(pm, 1, 1, cPitch - muPitch, gammaW, ...
    'sd', widthW, 'dropInputAttr', false);

fprintf('  T then W (centre c = %g):\n', cPitch);
fprintf('    pmPath1.wAttr{1} = [%s]\n', num2str(pmPath1.wAttr{1}, '%.4f '));
fprintf('  W (centre c - mu = %g) before T:\n', cPitch - muPitch);
fprintf('    pmPath2.wAttr{1} = [%s]\n', num2str(pmPath2.wAttr{1}, '%.4f '));
fprintf('  difference max = %g  (zero --- centre-shift rule holds)\n', ...
    max(abs(pmPath1.wAttr{1} - pmPath2.wAttr{1})));

%% ===================================================================
%  8b. transformAttributes: the measurement scale, and its order with D
%  ===================================================================

fprintf('\n=== 8b. transformAttributes (F): scale choice and order with D ===\n');

% The kernel of buildMaet has a fixed width in whatever units the
% values carry, so the choice of scale is made before the tensor. The
% bare-array form converts a vector in one call:
fHz     = [392.00 369.99 329.63];                 % G4, F#4, E4 in Hz
pCents  = transformAttributes(fHz, [], {'hz', 'cents'});
fprintf('  Hz -> cents: [%.1f %.1f %.1f]\n', pCents);

% A log scale turns uniform scaling -- a tempo change, or an
% augmentation of a melody's intervals -- into a translation, which the
% relative flag or a translation sweep can then absorb:
% demo_repetitionHandling and demo_tempoInvariance build on this.

% Order with differencing carries meaning. (i) F then D on inter-onset
% intervals in log2 gives log ratios: a doubling is +1, a halving -1.
ioi        = [0.25 0.5 0.5 1.0];                  % seconds
pmLD = differenceEvents( ...
    transformAttributes({ioi}, [], {{'log', 'base', 2}}), 1);
fprintf('  log2(IOI) then D: [%g %g %g]  (log ratios)\n', pmLD.pAttr{1});

% (ii) D then a compressive transform on the signed pitch intervals.
% log(x + 1) admits the zero of a repeated note with the constant written
% down. Negative values are refused unless a sign attribute is requested:
% with 'sign', true the transform is applied to |x| and a sign
% attribute at the 2-point simplex's vertices, {-1/2, 0, +1/2}, is
% inserted right after its source, so the pre-MAET grows from one
% attribute to two.
pmDp = differenceEvents(selectPreMaet(pm, 'attributes', 1), 1);
pmF  = transformAttributes(pmDp, {{'log', 'offset', 1}}, 'sign', true);
fprintf('  D(pitch)        = [%g %g]\n', pmDp.pAttr{1});
fprintf('  log(|D(pitch)|+1) = [%.4f %.4f], sign = [%g %g] (spec name ''%s'')\n', ...
        pmF.pAttr{1}, pmF.pAttr{2}, pmF.specs{2}.name);
% A log has no single image of the old width, so the rescaled attributes
% carry sigma as NA (demo_preMaetIo, section 6), and the widths the new
% units call for are supplied here.
densF = buildMaet(pmF, 'sigma', [0.2 0.3], ...
                     'isPer', [false false], 'period', [0 0], 'verbose', false);
fprintf('  buildMaet on the two-attribute pre-MAET: dim = %d\n', densF.dim);

% (iii) A zero under 'log' is an error with remedies, never -Inf.
try
    transformAttributes({[0.5 0 0.25]}, [], {'log'});
catch ME
    fprintf('  zero IOI under ''log'' -> %s\n', strtok(ME.message, ';'));
end

% (iv) A function handle is accepted alongside the named transforms.
pmUser = transformAttributes({[1 4 9]}, [], {@(x) sqrt(x) + 1});
fprintf('  user function sqrt(x) + 1: [%g %g %g]\n\n', pmUser.pAttr{1});

%% ===================================================================
%  9. The tensor functions take the pre-MAET whole
%  ===================================================================

fprintf('\n=== 9. Pre-MAET form: the tensor functions take the pre-MAET whole ===\n');

% entropyMaet, evalMaet, and simMaet each take a pre-MAET whole and read
% the kernel geometry from its specs; the density is built inside the
% call. This is the form to reach for first.

% --- 9a. entropyMaet ---
H_orig = entropyMaet(pm, 'method', 'renyi2', 'verbose', false);
fprintf('  entropyMaet(pm)\n');
fprintf('    = %.4f  (Renyi-2)\n', H_orig);

% The raw positional form, with the parts and the five geometry vectors
% (sigma, r, isRel, isPer, period) written out, reaches the same value;
% it serves data that was never packed as a pre-MAET.
[p0, w0] = unpackPreMaet(pm);
H_raw = entropyMaet(p0, w0, [0.5 0.25], [1 1], [false false], ...
                    [true false], [12 0], 'method', 'renyi2', ...
                    'verbose', false);
fprintf('  raw positional form: %.4f  (|delta| = %.2e)\n', ...
        H_raw, abs(H_raw - H_orig));
assert(abs(H_raw - H_orig) < 1e-12, ...
       'Section 9a: pre-MAET and raw forms disagree.');

% --- 9b. evalMaet at the penult event (pitch = 66, t = 6) ---
% Query points are A-by-M_q, one column per query and row a giving
% attribute a's value(s). Single query here, so a 2-by-1 column.
Xq = [66; 6];
val_at_penult = evalMaet(pm, Xq, 'verbose', false);
fprintf('  evalMaet(pm, Xq)\n');
fprintf('    = %.4f\n', val_at_penult);
fprintf('  (The density at an actual event: nearly all of it is event 6''s\n');
fprintf('   own kernel, its neighbours'' tails adding little.)\n');

% --- 9c. simMaet of the fragment and its transposed copy (Section 4) ---
% The pitch kernel is narrow (sigma = 0.5 semitones), so the 5-semitone
% shift puts every event out of kernel reach of its original pitch
% class, and the similarity collapses to nearly 0. Differencing, in 9d,
% recovers it.
sim_T = simMaet(pm, pmT, 'verbose', false);
fprintf('  simMaet(pm, pmT)\n');
fprintf('    = %.4f\n', sim_T);

% --- 9d. simMaet of the differenced pair: D o T == D in action ---
% Section 7's identity guarantees that the differenced original and the
% differenced transposed copy are identical in value, so their cosine
% similarity is exactly 1: the algebraic identity surfacing as a
% downstream observable.
sim_diffed = simMaet(pmD, pmDT, 'verbose', false);
fprintf('  simMaet(pmD, pmDT)\n');
fprintf('    = %.4f  (exactly 1: D absorbs T)\n', sim_diffed);

%% ===================================================================
%  10. Build once, query many
%  ===================================================================

fprintf('\n=== 10. Density form: build once, query many; parity with Section 9 ===\n');

% Where one density is evaluated or compared many times, build it once
% with buildMaet and pass the density instead: the structural work
% (canonical forms, tuple indices, weight products) is then paid once.
dens_orig = buildMaet(pm,   'verbose', false);
dens_T    = buildMaet(pmT,  'verbose', false);
dens_D    = buildMaet(pmD,  'verbose', false);
dens_DT   = buildMaet(pmDT, 'verbose', false);
dens_W    = buildMaet(pmW,  'verbose', false);

% --- 10a. entropyMaet on the density; same answer as 9a. ---
H_orig_dens = entropyMaet(dens_orig, 'method', 'renyi2', ...
                             'verbose', false);
fprintf('  entropyMaet(dens_orig)\n');
fprintf('    = %.4f  (Renyi-2; parity vs 9a: |delta| = %.2e)\n', ...
        H_orig_dens, abs(H_orig_dens - H_orig));
assert(abs(H_orig_dens - H_orig) < 1e-12, ...
       'Section 10a: entropy pre-MAET and density forms disagree.');

% --- 10b. evalMaet at the same query; same answer as 9b. ---
val_at_penult_dens = evalMaet(dens_orig, Xq, 'verbose', false);
fprintf('  evalMaet(dens_orig, Xq)\n');
fprintf('    = %.4f  (parity vs 9b: |delta| = %.2e)\n', ...
        val_at_penult_dens, abs(val_at_penult_dens - val_at_penult));
assert(abs(val_at_penult_dens - val_at_penult) < 1e-12, ...
       'Section 10b: eval pre-MAET and density forms disagree.');

% --- 10c. simMaet(dens_orig, dens_T); same answer as 9c. ---
sim_T_dens = simMaet(dens_orig, dens_T, 'verbose', false);
fprintf('  simMaet(dens_orig, dens_T)\n');
fprintf('    = %.4f  (parity vs 9c: |delta| = %.2e)\n', ...
        sim_T_dens, abs(sim_T_dens - sim_T));
assert(abs(sim_T_dens - sim_T) < 1e-12, ...
       'Section 10c: simMaet pre-MAET and density forms disagree.');

% --- 10d. simMaet(dens_D, dens_DT) on the differenced pair; ---
%       same answer as 9d. (Section 7 identity: exactly 1.)
sim_diffed_dens = simMaet(dens_D, dens_DT, 'verbose', false);
fprintf('  simMaet(dens_D, dens_DT)\n');
fprintf('    = %.4f  (parity vs 9d: |delta| = %.2e)\n', ...
        sim_diffed_dens, abs(sim_diffed_dens - sim_diffed));
assert(abs(sim_diffed_dens - sim_diffed) < 1e-12, ...
       'Section 10d: simMaet pre-MAET and density forms disagree.');

% --- 10e. List form: one reference against many candidates. ---
% A cell of densities against one density returns one similarity per
% entry, for "compare one reference against many" workflows. Three
% entries, {dens_orig, dens_T, dens_W}:
%   entry 1:  sim(orig, orig) = 1 by definition.
%   entry 2:  sim(orig, T), which matches 9c.
%   entry 3:  sim(orig, W), the same values under the cadence window of
%             Section 5, so only the weights differ.
% demo_batchProcessing takes the list forms further, on a table of
% trials.
sim_list = simMaet({dens_orig, dens_T, dens_W}, dens_orig, ...
                   'verbose', false);
fprintf('  simMaet({dens_orig, dens_T, dens_W}, dens_orig)\n');
fprintf('    = {%.4f, %.4f, %.4f}\n', sim_list{1}, sim_list{2}, sim_list{3});
fprintf('    (entry 1: self = 1; entry 2: vs T (= 9c); entry 3: vs W, the\n');
fprintf('     same values under the cadence window, so only the weights differ.)\n');
assert(abs(sim_list{1} - 1)     < 1e-12, '10e: self-similarity not 1.');
assert(abs(sim_list{2} - sim_T) < 1e-12, '10e: list entry 2 != 9c value.');

mptDefaults(prevDefaults);
