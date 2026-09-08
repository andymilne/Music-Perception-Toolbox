%% demo_preprocessing.m
% Pre-MAET preprocessing operations and their compositions.
%
% Demonstrates the pre-MAET preprocessing helpers in MPT, applied to a
% fragment of J. S. Bach, BWV 347 ("Ich dank dir, lieber Herre"): the
% soprano over quarter-notes t = 1 to 7, which is bar 1 entire followed
% by cadence 1's three-chord approach (antepenult i, penult V, tonic I
% at t = 5, 6, 7), so the fragment ends on its cadential goal. Two
% attributes are kept: the soprano pitch (attribute 1, treated as
% periodic mod 12 so it lives on the pitch-class circle) and the event
% time in quarter-notes (attribute 2, non-periodic). Each subsequent
% section illustrates one operation or one composition; the operations
% leave the source pAttr untouched.
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
%       differenceEvents  (D): per-attribute difference orders.
%       bindEvents        (B): per-attribute bind orders (n-gram
%                              expansion).
%       translateAttributes (T): per-attribute translation of values.
%       weightEvents      (W): per-event window (one factor per
%                              input attribute, peak-normalised
%                              fixed-variance family).
%       transformAttributes (F): per-attribute elementwise maps
%                              (log, scale conversion, user function)
%                              and the sign attribute.
%
%   Compositions
%       D o B == B o D    (n-tuple entropy pipeline commutation,
%                          value-wise and weight-wise after attribute
%                          permutation).
%       D o T == D        (differencing absorbs absolute translation;
%                          T o D adds mu to every difference).
%       T o W centre shift (W with centre c after T(mu) equals W with
%                          centre c - mu before T; W leaves T
%                          invariant on values).
%       D o F vs F o D    (log then difference gives log ratios;
%                          difference then log(x+1) + sign gives signed
%                          compressed magnitudes).
%
% The Python mirror is demos/demo_preprocessing.py.
%
% See also SHOWPREMAET, DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES,
% WEIGHTEVENTS, TRANSFORMATTRIBUTES.

clear; clc;

% Shown on every table, so that the parameters that would build the
% density travel with the values they would be built from.
KERNEL = {'sigma', [0.5 0.25], 'isPer', [true false], ...
          'period', [12 0], 'names', {'pitch', 'time'}};

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

% The three parts travel together as one pre-MAET, which every operator
% below takes whole and returns whole.
pm = preMaet(pAttr, w);

isRel   = [false false];      % both attributes are absolute
isPer   = [true  false];      % attribute 1 is periodic (PC), attribute 2 isn't
periods = [12 0];             % period 12 (semitones) for PC

showPreMaet(pm, KERNEL{:});
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
% on the undifferenced one.
diffOrders = [1 0];
pmD = differenceEvents(pm, diffOrders);

fprintf('  diffOrders = [%d %d]   (pitch differenced, time left alone)\n', ...
    diffOrders(1), diffOrders(2));
showPreMaet(pmD, KERNEL{:}, 'format', 'latex');
fprintf('\n');

%% ===================================================================
%  3. bindEvents (B): expand attributes into 2-grams
%  ===================================================================

fprintf('=== 3. bindEvents (B) ===\n');

% Bind 2 consecutive events into 2-grams on both attributes. Each source
% attribute becomes ONE nested attribute: the two bound events form the
% ordered outer level, each event's own value the inner level (inheriting
% the source r/isRel/isSym). A' = A = 2. Trailing-drop alignment gives
% N' = N - max(L) + 1 = 6. B gathers each super-event's constituent
% weights alongside its values rather than combining them, so both of a
% 2-gram's metre weights survive, in order, inside the cell.
bindOrders = [2 2];
pmB = bindEvents(pm, bindOrders);
specB = pmB.specs;

fprintf(['  bindOrders = [%d %d]   (A'' = %d: each source attribute ' ...
    '-> one nested attribute)\n'], bindOrders(1), bindOrders(2), ...
    numel(pmB.pAttr));
showPreMaet(pmB, KERNEL{:});
fprintf('  spec(1): r = [%s], sym = [%s], rel = [%s], tags = [%s]\n', ...
    num2str(specB{1}.r), num2str(double(specB{1}.sym)), ...
    num2str(double(specB{1}.rel)), num2str(specB{1}.tags(:)'));
fprintf('\n');

%% ===================================================================
%  3b. bindEvents again (B o B): deepen a nested attribute to L = 3
%  ===================================================================

fprintf('=== 3b. bindEvents again (B o B): deepen to L = 3 ===\n');

% bindEvents accepts the pre-MAET it produces, so a second bind deepens
% the *already-nested* attribute rather than starting over. The specs
% travel with the pre-MAET, so nothing has to be threaded by hand. The
% existing tag matrix is tiled and a fresh outermost grouping
% column is appended; r/sym/rel each gain one outer level. The hierarchy
% grows note -> 2-event group (first bind) -> 2-group window (second
% bind). Trailing-drop again: N'' = N' - max(L) + 1 = 5.
pmBB = bindEvents(pmB, [2 2]);
specBB = pmBB.specs;

fprintf('  N'''' = %d  (three-level super-events)\n', size(pmBB.pAttr{1}, 2));
showPreMaet(pmBB, 'maxEvents', 4, KERNEL{:});
fprintf('  spec(1): r = [%s], sym = [%s], rel = [%s]\n', ...
    num2str(specBB{1}.r), num2str(double(specBB{1}.sym)), ...
    num2str(double(specBB{1}.rel)));
fprintf('  (inner tag column tiled; a new outermost column appended.)\n');

% Build the absolute L=3 nest and confirm a clean self-similarity. The
% specs here carry no kernel geometry, so these arguments supply it; where
% a spec does carry a value, an argument overrides it instead, which is
% what makes a sweep one call per value (demo_preMaetIo, section 4).
kwBB = {'sigma', [0.5 0.25], 'isPer', [true false], ...
        'period', [12 0], 'verbose', false};
dBBAbs = buildExpTens(pmBB, kwBB{:});
smAbs = cosSimExpTens(dBBAbs, dBBAbs, 'verbose', false);
fprintf('  absolute build: dim = %d, cosine self-match = %.4f\n', ...
    dBBAbs.dim, smAbs);

% Outermost [rel] on pitch quotients the whole 3-level tuple by a common
% shift: the doubly-bound pitch structure is then invariant to transposing
% every note together. Absolute (no [rel]) is not --- the narrow PC kernel
% (sigma = 0.5) puts a 5-semitone shift out of reach.
% Replacing one part of a pre-MAET leaves the rest in place: here the
% specs, and below the values.
specBBOut = specBB;
specBBOut{1}.rel = [0 0 1];                  % outermost unit on pitch
pmBBOut = preMaet(pmBB, [], specBBOut);
dBBOut = buildExpTens(pmBBOut, kwBB{:});
pmBBT = pmBB;
pmBBT.pAttr{1} = pmBB.pAttr{1} + 5;          % transpose all pitches +5
pmBBOutT = preMaet(pmBBT, [], specBBOut);
dBBOutT = buildExpTens(pmBBOutT, kwBB{:});
dBBAbsT = buildExpTens(pmBBT, kwBB{:});
simOut = cosSimExpTens(dBBOut, dBBOutT, 'verbose', false);
simAbs = cosSimExpTens(dBBAbs, dBBAbsT, 'verbose', false);
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
% attribute's values (here K=1 each). isRel is read from specs
% (synthesised flat: both absolute), so neither is a no-op. T moves
% values only: the weights below are the metre weights unchanged.
muPitch = 5;
mu = {muPitch, 0};
pmT = translateAttributes(pm, mu);

fprintf('  mu (per attribute) = {%g, %g}   (G->C, F#->B, E->A; time untouched)\n', ...
    mu{1}, mu{2});
showPreMaet(pmT, KERNEL{:});
fprintf('\n');

%% ===================================================================
%  5. weightEvents (W): window the time attribute at the cadence
%  ===================================================================

fprintf('=== 5. weightEvents (W) ===\n');

% Apply a window on the time axis (input attribute 2) centred at the
% penult event (t = 6) with standard deviation 2 quarter-notes and
% gamma = 0 (pure Gaussian). The factor lands back on the time attribute
% (target attribute 2), the in-place weighting case, and the input is
% kept (dropInputAttr = false). The window multiplies the metre weights
% it finds rather than replacing them, so the time row below carries
% metre times envelope, and the pitch row is untouched.
pmW = weightEvents(pm, 2, 2, 6, 0, 'sd', 2, 'dropInputAttr', false);

fprintf(['  inputAttr = 2 (time); targetAttr = 2; centre = 6; sd = 2; ' ...
    'shape = 0 (Gaussian)\n']);
showPreMaet(pmW, 'decimals', 3, KERNEL{:});
fprintf('\n');

%% ===================================================================
%  6. D o B == B o D (n-tuple entropy pipeline commutation)
%  ===================================================================

fprintf('=== 6. B o D == D o B (pipeline commutation) ===\n');

% Both pre-MAET operators take a pre-MAET and return one, so they compose
% directly and the two routes coincide. Differencing pairs values position by position across
% (super-)events and the sliding bind window commutes with it, on the
% ordered/K=1 domain where difference is defined. The two operators
% propagate weights by different rules --- D takes the rolling product,
% B gathers --- and the composition agrees on the weights as well.
%   D then B: difference each attribute (order 1), then bind 2-grams.
pmDB = bindEvents(differenceEvents(pm, [1 1]), [2 2]);
%   B then D: bind 2-grams, then difference each nested attribute
%   position by position.
pmBD = differenceEvents(bindEvents(pm, [2 2]), [1 1]);

showPreMaet(pmDB, 'title', '  D then B:', KERNEL{:});
showPreMaet(pmBD, 'title', '  B then D:', KERNEL{:});

valsAgree = true; wtsAgree = true; specsAgree = true;
for a = 1:2
    valsAgree = valsAgree && isequaln(pmDB.pAttr{a}, pmBD.pAttr{a});
    wtsAgree = wtsAgree && isequaln(pmDB.wAttr{a}, pmBD.wAttr{a});
    for f = {'tags', 'r', 'sym', 'rel'}
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

showPreMaet(pmDT, 'title', '  D(T(p)):', KERNEL{:});
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
wPath1 = pmPath1.wAttr;

% Path 2: W centred at c - mu = 64 BEFORE T (T leaves weights untouched).
pmPath2 = weightEvents(pm, 1, 1, cPitch - muPitch, gammaW, ...
    'sd', widthW, 'dropInputAttr', false);
wPath2 = pmPath2.wAttr;

fprintf('  T then W (centre c = %g):\n', cPitch);
fprintf('    wPath1{1} = [%s]\n', num2str(wPath1{1}, '%.4f '));
fprintf('  W (centre c - mu = %g) before T:\n', cPitch - muPitch);
fprintf('    wPath2{1} = [%s]\n', num2str(wPath2{1}, '%.4f '));
fprintf('  difference max = %g  (zero --- centre-shift rule holds)\n', ...
    max(abs(wPath1{1} - wPath2{1})));

%% ===================================================================
%  8b. transformAttributes: the measurement scale, and its order with D
%  ===================================================================

fprintf('\n=== 8b. transformAttributes (F): scale choice and order with D ===\n');

% The kernel of buildExpTens has a fixed width in whatever units the
% values carry, so the choice of scale is made before the tensor. The
% bare-array form converts a vector in one call (this replaces the
% former convertPitch):
fHz     = [392.00 369.99 329.63];                 % G4, F#4, E4 in Hz
pCents  = transformAttributes(fHz, [], {'hz', 'cents'});
fprintf('  Hz -> cents: [%.1f %.1f %.1f]\n', pCents);

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
% inserted right after its source, so the
% pre-MAET grows from one attribute to two (note the two sigmas below).
pmDp = differenceEvents(pAttr(1), w(1), 1);
pmF  = transformAttributes(pmDp, {{'log', 'offset', 1}}, 'sign', true);
fprintf('  D(pitch)        = [%g %g]\n', pmDp.pAttr{1});
fprintf('  log(|D(pitch)|+1) = [%.4f %.4f], sign = [%g %g] (spec name ''%s'')\n', ...
        pmF.pAttr{1}, pmF.pAttr{2}, pmF.specs{2}.name);
densF = buildExpTens(pmF, 'sigma', [0.2 0.3], ...
                     'isPer', [false false], 'period', [0 0], 'verbose', false);
fprintf('  buildExpTens on the two-attribute pre-MAET: dim = %d\n', densF.dim);

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
%  9. Pre-MAET into the raw multi-attribute form (route (ii))
%  ===================================================================

fprintf('\n=== 9. Raw form: pre-MAET feeds directly into tensor functions ===\n');

% Two routes lead from a pre-MAET to a density value, an entropy, or a
% similarity:
%
%   (i)  build a MaetDensity once via buildExpTens, then pass the
%        struct to entropyExpTens / evalExpTens / cosSimExpTens.
%        Preferred when the same density is re-evaluated many times,
%        because the structural work (group canonicalisation, tuple
%        index pre-computation, weight products) is paid once.
%
%   (ii) call the raw multi-attribute form of each function directly,
%        passing (pAttr, wAttr, sigma, r, isRel, isPer, periods)
%        as positional arguments. The function builds the density
%        internally and returns the answer; no struct is exposed.
%        Convenient for single-shot uses and keeps the call shape
%        symmetric with buildExpTens itself.
%
% Section 9 below exercises route (ii) on the original pAttr and on
% the differenced / translated pre-MAETs. Section 10 then exercises
% route (i) on the same set, building each density once via
% buildExpTens and reusing it across entropyExpTens, evalExpTens,
% cosSimExpTens, and the LIST form of cosSimExpTens, with parity
% assertions confirming the two routes return identical values.
sigma = [0.5, 0.25];     % kernel std: 0.5 semitones (PC), 0.25 quarter-notes (time)
r     = [1, 1];          % single-value attributes (K_a = 1)

% The raw form takes the parts positionally, so the pre-MAETs above are
% read out into the cells it expects.
pT = pmT.pAttr;
pD = pmD.pAttr;
wD = pmD.wAttr;

% --- 9a. entropyExpTens (raw MA form) ---
% Signature:
%   H = entropyExpTens(pAttr, w, sigma, r, isRel, isPer, periods, ...)
H_orig = entropyExpTens(pAttr, w, sigma, r, isRel, isPer, periods, ...
                        'method', 'renyi2', ...
                        'verbose', false);
fprintf('  entropyExpTens(pAttr, w, sigma, r, isRel, isPer, periods)\n');
fprintf('    = %.4f  (Renyi-2)\n', H_orig);

% --- 9b. evalExpTens at the penult event (pitch = 66, t = 6) ---
% Signature:
%   vals = evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, X, ...)
% Query points are A-by-M_q with one column per query and row a
% giving attribute a's value(s). Single query here, so a 2-by-1 column.
Xq = [66; 6];
val_at_penult = evalExpTens(pAttr, w, sigma, r, ...
                            isRel, isPer, periods, Xq, ...
                            'verbose', false);
fprintf('  evalExpTens(pAttr, w, sigma, r, isRel, isPer, periods, Xq)\n');
fprintf('    = %.4f\n', val_at_penult);
fprintf('  (Density peak near an actual event; the value reflects the\n');
fprintf('   contribution from event 6 at (66, 6) plus tails from its neighbours.)\n');

% --- 9c. cosSimExpTens on two pre-MAETs (raw MA form) ---
% Signature:
%   s = cosSimExpTens(pX, wX, pY, wY, sigma, r, isRel, isPer, periods, ...)
% Compare the original chorale fragment against the transposed copy
% (Section 4). Group 1's PC kernel is narrow (sigma = 0.5 semitones),
% so the 5-semitone shift puts every event out of kernel reach of its
% original PC, and the similarity collapses to 0. Pre-MAET D in step
% 9d below recovers it.
sim_T = cosSimExpTens(pAttr, w, pT, w, sigma, r, ...
                      isRel, isPer, periods, 'verbose', false);
fprintf('  cosSimExpTens(pAttr, w, pT, w, sigma, r, isRel, isPer, periods)\n');
fprintf('    = %.4f\n', sim_T);

% --- 9d. cosSim of the differenced pair: D(T) == D identity in action ---
% Section 7's identity D o T == D guarantees that the differenced
% original and the differenced transposed copy are value-wise
% identical, so their cosine similarity must be exactly 1. The
% algebraic identity from Section 7 surfacing as a downstream
% observable; no buildExpTens required.
sim_diffed = cosSimExpTens(pD, wD, pmDT.pAttr, pmDT.wAttr, ...
                           sigma, r, isRel, isPer, periods, ...
                           'verbose', false);
fprintf('  cosSimExpTens(pD, wD, pD(T), wD(T), ...)\n');
fprintf('    = %.4f  (exactly 1: D absorbs T)\n', sim_diffed);

%% ===================================================================
%  10. Pre-MAET via buildExpTens dens structs (route (i))
%  ===================================================================

fprintf('\n=== 10. Dens form: build once, query many; parity with route (ii) ===\n');

% Build each pre-MAET into a MaetDensity struct once. After this the
% structural work --- group canonicalisation, tuple-index
% pre-computation, weight products --- is paid; subsequent
% entropy/eval/cosSim calls just consume the struct.
dens_orig = buildExpTens(pAttr, w, sigma, r, ...
                         isRel, isPer, periods, 'verbose', false);
dens_T    = buildExpTens(pT,    w, sigma, r, ...
                         isRel, isPer, periods, 'verbose', false);
dens_D    = buildExpTens(pD,   wD, sigma, r, ...
                         isRel, isPer, periods, 'verbose', false);
dens_DT   = buildExpTens(pmDT.pAttr, pmDT.wAttr, sigma, r, ...
                         isRel, isPer, periods, 'verbose', false);

% --- 10a. entropyExpTens on the struct; same answer as 9a. ---
H_orig_dens = entropyExpTens(dens_orig, 'method', 'renyi2', ...
                             'verbose', false);
fprintf('  entropyExpTens(dens_orig)\n');
fprintf('    = %.4f  (Renyi-2; parity vs 9a: |delta| = %.2e)\n', ...
        H_orig_dens, abs(H_orig_dens - H_orig));
assert(abs(H_orig_dens - H_orig) < 1e-12, ...
       'Section 10a: entropy raw and dens forms disagree.');

% --- 10b. evalExpTens at the same query; same answer as 9b. ---
val_at_penult_dens = evalExpTens(dens_orig, Xq, 'verbose', false);
fprintf('  evalExpTens(dens_orig, Xq)\n');
fprintf('    = %.4f  (parity vs 9b: |delta| = %.2e)\n', ...
        val_at_penult_dens, abs(val_at_penult_dens - val_at_penult));
assert(abs(val_at_penult_dens - val_at_penult) < 1e-12, ...
       'Section 10b: eval raw and dens forms disagree.');

% --- 10c. cosSimExpTens(dens_orig, dens_T); same answer as 9c. ---
sim_T_dens = cosSimExpTens(dens_orig, dens_T, 'verbose', false);
fprintf('  cosSimExpTens(dens_orig, dens_T)\n');
fprintf('    = %.4f  (parity vs 9c: |delta| = %.2e)\n', ...
        sim_T_dens, abs(sim_T_dens - sim_T));
assert(abs(sim_T_dens - sim_T) < 1e-12, ...
       'Section 10c: cosSim raw and dens forms disagree.');

% --- 10d. cosSimExpTens(dens_D, dens_DT) on the differenced pair; ---
%       same answer as 9d. (Section 7 identity: should be exactly 1.)
sim_diffed_dens = cosSimExpTens(dens_D, dens_DT, 'verbose', false);
fprintf('  cosSimExpTens(dens_D, dens_DT)\n');
fprintf('    = %.4f  (parity vs 9d: |delta| = %.2e)\n', ...
        sim_diffed_dens, abs(sim_diffed_dens - sim_diffed));
assert(abs(sim_diffed_dens - sim_diffed) < 1e-12, ...
       'Section 10d: cosSim raw and dens forms disagree.');

% --- 10e. LIST form: one reference against many candidates. ---
% Scalar-vs-list cosSimExpTens broadcasts dens_orig against each
% candidate in the cell, returning a 1-by-n cell of similarity
% scalars. Useful for "compare one reference density against many"
% workflows.
%
% Four entries are returned for {dens_orig, dens_T, dens_D, dens_DT}:
%   entry 1:  sim(orig, orig) = 1 by definition.
%   entry 2:  sim(orig, T) --- matches 9c's sim_T.
%   entry 3:  sim(orig, D(orig)) --- new value; how similar the
%             original pAttr is to its first-difference.
%   entry 4:  sim(orig, D(T))   --- Section 7's identity D o T == D
%             forces this to equal entry 3.
sim_list = cosSimExpTens({dens_orig, dens_T, dens_D, dens_DT}, dens_orig, ...
                         'verbose', false);
fprintf('  cosSimExpTens({dens_orig, dens_T, dens_D, dens_DT}, dens_orig)\n');
fprintf('    = {%.4f, %.4f, %.4f, %.4f}\n', ...
        sim_list{1}, sim_list{2}, sim_list{3}, sim_list{4});
fprintf('    (entry 1: self = 1; entry 2: vs T (= 9c);\n');
fprintf('     entry 3: vs D(orig); entry 4: vs D(T) --- equals entry 3 by D o T == D.)\n');
assert(abs(sim_list{1} - 1)              < 1e-12, '10e: self-similarity not 1.');
assert(abs(sim_list{2} - sim_T)          < 1e-12, '10e: LIST entry 2 != 9c value.');
assert(abs(sim_list{3} - sim_list{4})    < 1e-12, ...
       '10e: D o T == D identity violated (entries 3 and 4 should match).');
