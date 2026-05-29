%DEMO_PREPROCESSING  Pre-MAET preprocessing operations and their compositions.
%
% Demonstrates the four per-event preprocessing helpers in MPT,
% applied to a small fragment of J. S. Bach, BWV 347 ("Ich dank dir,
% lieber Herre"). The fragment is cadence 1's three-chord approach
% (antepenult i, penult V, tonic I, at quarter-note positions
% t = 5, 6, 7) reduced to the soprano line for clarity. Two
% attributes are kept: the soprano pitch (group 1, treated as
% periodic mod 12 so it lives on the pitch-class circle) and the
% event time in quarter-notes (group 2, non-periodic). Each
% subsequent section illustrates one operation or one composition;
% the operations leave the source pAttr untouched.
%
%   Operations
%       differenceEvents  (D): per-attribute difference orders.
%       bindEvents        (B): per-attribute bind orders (n-gram
%                              expansion).
%       translateAttributes (T): per-group translation of values.
%       weightEvents      (W): per-event window via Design P (one
%                              factor per input attribute, peak-
%                              normalised fixed-variance family).
%
%   Compositions
%       D o B == B o D    (n-tuple entropy pipeline commutation,
%                          value-wise after attribute permutation).
%       D o T == D        (differencing absorbs absolute translation;
%                          T o D adds mu to every difference).
%       T o W centre shift (W with centre c after T(mu) equals W with
%                          centre c - mu before T; W leaves T
%                          invariant on values).
%
% The Python mirror is demos/demo_preprocessing.py.
%
% See also DIFFERENCEEVENTS, BINDEVENTS, TRANSLATEATTRIBUTES, WEIGHTEVENTS.

clear; clc;

%% ===================================================================
%  1. BWV 347 cadence 1 input (soprano + time)
%  ===================================================================

fprintf('=== 1. Inputs (BWV 347 cadence 1 soprano, three-chord approach) ===\n');

% Two attributes, both K_a = 1, three events.
pAttr   = { [67 66 64]; ...    % a_1: soprano pitch (G4, F#4, E4)
            [ 5  6  7] };      % a_2: event time in quarter-notes
w       = [];                  % weights: default (uniform).
groups  = [1 2];               % attribute a -> group g(a)
isRel   = [false false];       % both groups are absolute
isPer   = [true  false];       % group 1 is periodic (PC), group 2 isn't
periods = [12 0];              % period 12 (semitones) for PC

fprintf('  pitch (a_1):  [%g %g %g]\n', pAttr{1});
fprintf('  time  (a_2):  [%g %g %g]\n', pAttr{2});
fprintf('  group 1 (PC):  periodic, P = 12\n');
fprintf('  group 2 (time): non-periodic\n\n');


%% ===================================================================
%  2. differenceEvents (D): turn pitches into intervals
%  ===================================================================

fprintf('=== 2. differenceEvents (D) ===\n');

% Take the first difference of pitch and leave time alone. The first
% event is dropped (leading-drop alignment): N' = N - max(k) = 2.
diffOrders   = {1, 0};
[pD, wD, gD] = differenceEvents(pAttr, w, groups, diffOrders);

fprintf('  diffOrders   = {1, 0}\n');
fprintf('  D(pitch)     = [%g %g]   (interval sequence: F#-G, E-F#)\n', pD{1});
fprintf('  D(time)      = [%g %g]   (unchanged in value, leading event dropped)\n\n', pD{2});


%% ===================================================================
%  3. bindEvents (B): expand attributes into 2-grams
%  ===================================================================

fprintf('=== 3. bindEvents (B) ===\n');

% Bind 2 consecutive events into 2-grams on both attributes. Each
% source attribute yields L_a = 2 super-attributes (the two slots of
% the 2-gram), so A' = sum_a L_a = 4. Trailing-drop alignment gives
% N' = N - max(L) + 1 = 2.
bindOrders   = {2, 2};
[pB, wB, gB] = bindEvents(pAttr, w, groups, bindOrders);

fprintf('  bindOrders   = {2, 2}\n');
fprintf('  A'' = %d (each source attribute expands to L_a = 2 super-attributes)\n', numel(pB));
fprintf('  pitch 2-grams: a_1 slot 1 = [%g %g]; a_1 slot 2 = [%g %g]\n', ...
        pB{1}, pB{2});
fprintf('  time  2-grams: a_2 slot 1 = [%g %g]; a_2 slot 2 = [%g %g]\n\n', ...
        pB{3}, pB{4});


%% ===================================================================
%  4. translateAttributes (T): transpose pitch up a perfect fourth
%  ===================================================================

fprintf('=== 4. translateAttributes (T) ===\n');

% Translate pitch (group 1) by +5 semitones; leave time alone.
% Cell form keeps it a single translation: {scalar_group1, scalar_group2}.
% A length-G numeric vector would instead be read as a 2-position sweep
% (broadcast across attributes) under the orientation grammar.
muPitch = 5;
pT = translateAttributes(pAttr, groups, {muPitch, 0}, isRel, isPer, periods);

fprintf('  mu (per group) = {%g, %g}   (group 1: pitch; group 2: time)\n', muPitch, 0);
fprintf('  T(pitch)       = [%g %g %g]   (G->C, F#->B, E->A)\n', pT{1});
fprintf('  T(time)        = [%g %g %g]   (unchanged)\n\n', pT{2});


%% ===================================================================
%  5. weightEvents (W): window the time attribute at the penult
%  ===================================================================

fprintf('=== 5. weightEvents (W) ===\n');

% Apply a window on the time attribute (input = 2) centred at the penult
% event (t = 6) with standard deviation 1 quarter-note and gamma = 0
% (pure Gaussian). The factor lands back on the time slot itself
% (target = 2), which is the in-place weighting use case.
[~, wOut, ~] = weightEvents(pAttr, w, groups, ...
                            2, ...        % inputAttr: time
                            2, ...        % targetAttr: time (in-place)
                            6, ...        % centre at t = 6 (the penult)
                            1, ...        % window std = 1 quarter-note
                            0, ...        % gamma = 0 -> pure Gaussian
                            false, 0, ...
                            'deleteInput', false);

fprintf('  inputAttr = 2 (time); targetAttr = 2; centre = 6; width = 1; shape (gamma) = 0\n');
fprintf('  wOut{1} (pitch, untouched): [%s]\n', mat2str(wOut{1}));
fprintf('  wOut{2} (time, windowed):   [%g %g %g]\n', wOut{2});
fprintf('  (peak at t = 6; falls off symmetrically by exp(-(t-6)^2/2).)\n\n');


%% ===================================================================
%  6. D o B == B o D (n-tuple entropy pipeline commutation)
%  ===================================================================

fprintf('=== 6. D o B == B o D ===\n');

% Path 1: D then B (difference first, then bind into 2-grams).
[pD1, wD1, gD1]   = differenceEvents(pAttr, w, groups, {1, 0});
[pDB, wDB, gDB]   = bindEvents(pD1, wD1, gD1, {2, 2});

% Path 2: B then D (bind first, then difference each super-attribute).
[pB1, wB1, gB1]   = bindEvents(pAttr, w, groups, {2, 2});
% After B, A' = 4 but G is still 2 (super-attributes inherit their
% source group). Difference group 1 (pitch slots) and leave group 2
% (time slots) alone --- the per-group form is the natural spelling.
[pBD, wBD, gBD]   = differenceEvents(pB1, wB1, gB1, {1, 0});

% Equality holds value-wise after recognising the cell-position
% reorder: D o B's two pitch slots equal B o D's two pitch slots, and
% likewise for time.
maxDiffPitch = max(abs([pDB{1} pDB{2}] - [pBD{1} pBD{2}]));
maxDiffTime  = max(abs([pDB{3} pDB{4}] - [pBD{3} pBD{4}]));
fprintf('  pitch slots agree to max |.| = %g\n', maxDiffPitch);
fprintf('  time  slots agree to max |.| = %g\n', maxDiffTime);
fprintf('  (Both routes yield the same value-wise output --- this is the\n');
fprintf('   commutation property used by the n-tuple entropy pipeline.)\n\n');


%% ===================================================================
%  7. D o T == D (differencing absorbs absolute translation)
%  ===================================================================

fprintf('=== 7. D o T == D ===\n');

% Difference applied to a transposed copy returns the same intervals
% as differencing the original: translation is wiped out by the
% difference operator (T o D, by contrast, adds mu to every
% difference).
pT_for_D            = translateAttributes(pAttr, groups, {muPitch, 0}, isRel, isPer, periods);
[pDT, wDT, gDT]     = differenceEvents(pT_for_D, w, groups, {1, 0});

fprintf('  D(T(pitch))    = [%g %g]\n', pDT{1});
fprintf('  D(pitch)       = [%g %g]\n', pD{1});
fprintf('  difference max = %g  (zero --- translation absorbed)\n\n', ...
        max(abs(pDT{1} - pD{1})));


%% ===================================================================
%  8. T o W centre-shift rule
%  ===================================================================

fprintf('=== 8. T o W centre shift ===\n');

% Path 1: T(mu) first (transposing pitch by +5), then W centred at
% the original pitch c = 67 (G4).
mu_pitch = 5;
c_pitch  = 67;
width_w  = 2.0;
gamma_w  = 0.3;
pT_path  = translateAttributes(pAttr, groups, {mu_pitch, 0}, isRel, isPer, periods);
[~, wPath1, ~] = weightEvents(pT_path, w, groups, ...
                              1, 1, ...           % input = target = pitch
                              c_pitch, width_w, gamma_w, ...
                              false, 0, ...
                              'deleteInput', false);

% Path 2: W centred at c - mu = 62 BEFORE T (T leaves weights
% untouched).
[~, wPath2, ~] = weightEvents(pAttr, w, groups, ...
                              1, 1, ...           % input = target = pitch
                              c_pitch - mu_pitch, width_w, gamma_w, ...
                              false, 0, ...
                              'deleteInput', false);

fprintf('  T then W (centre c = %g):\n    wPath1{1} = [%g %g %g]\n', c_pitch, wPath1{1});
fprintf('  W (centre c - mu = %g) before T:\n    wPath2{1} = [%g %g %g]\n', ...
        c_pitch - mu_pitch, wPath2{1});
fprintf('  difference max = %g  (zero --- centre-shift rule holds)\n', ...
        max(abs(wPath1{1} - wPath2{1})));

%% ===================================================================
%  9. Pre-MAET into the raw multi-attribute form (route (ii))
%  ===================================================================

fprintf('\n=== 9. Raw form: pre-MAET feeds directly into tensor functions ===\n');

% Two routes lead from a pre-MAET triple to a density value, an
% entropy, or a similarity:
%
%   (i)  build a MaetDensity once via buildExpTens, then pass the
%        struct to entropyExpTens / evalExpTens / cosSimExpTens.
%        Preferred when the same density is re-evaluated many times,
%        because the structural work (group canonicalisation, tuple
%        index pre-computation, weight products) is paid once.
%
%   (ii) call the raw multi-attribute form of each function directly,
%        passing (pAttr, w, sigma, r, groups, isRel, isPer, periods)
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
r     = [1, 1];          % single-slot attributes (K_a = 1) in both groups

% --- 9a. entropyExpTens (raw MA form) ---
% Signature:
%   H = entropyExpTens(pAttr, w, sigma, r, groups, isRel, isPer, periods, ...)
H_orig = entropyExpTens(pAttr, w, sigma, r, groups, isRel, isPer, periods, ...
                        'method', 'renyi2', 'normalize', false, ...
                        'verbose', false);
fprintf('  entropyExpTens(pAttr, w, sigma, r, groups, isRel, isPer, periods)\n');
fprintf('    = %.4f  (Renyi-2)\n', H_orig);

% --- 9b. evalExpTens at the penult event (pitch = 66, t = 6) ---
% Signature:
%   vals = evalExpTens(pAttr, w, sigma, r, groups, isRel, isPer, periods, X, ...)
% Query points are A-by-M_q with one column per query and row a
% giving attribute a's value(s). Single query here, so a 2-by-1 column.
Xq = [66; 6];
val_at_penult = evalExpTens(pAttr, w, sigma, r, groups, ...
                            isRel, isPer, periods, Xq, ...
                            'verbose', false);
fprintf('  evalExpTens(pAttr, w, sigma, r, groups, isRel, isPer, periods, Xq)\n');
fprintf('    = %.4f\n', val_at_penult);
fprintf('  (Density peak near an actual event; the value reflects the\n');
fprintf('   contribution from event 2 at (66, 6) plus tails from its neighbours.)\n');

% --- 9c. cosSimExpTens on two pre-MAETs (raw MA form) ---
% Signature:
%   s = cosSimExpTens(pX, wX, pY, wY, sigma, r, groups, isRel, isPer, periods, ...)
% Compare the original chorale fragment against the transposed copy
% (Section 4). Group 1's PC kernel is narrow (sigma = 0.5 semitones),
% so the 5-semitone shift puts every event out of kernel reach of its
% original PC, and the similarity collapses to 0. Pre-MAET D in step
% 9d below recovers it.
sim_T = cosSimExpTens(pAttr, w, pT, w, sigma, r, groups, ...
                      isRel, isPer, periods, 'verbose', false);
fprintf('  cosSimExpTens(pAttr, w, pT, w, sigma, r, groups, isRel, isPer, periods)\n');
fprintf('    = %.4f\n', sim_T);

% --- 9d. cosSim of the differenced pair: D(T) == D identity in action ---
% Section 7's identity D o T == D guarantees that the differenced
% original and the differenced transposed copy are value-wise
% identical, so their cosine similarity must be exactly 1. The
% algebraic identity from Section 7 surfacing as a downstream
% observable; no buildExpTens required.
[pDT_again, wDT_again, gDT_again] = differenceEvents(pT, w, groups, {1, 0});
sim_diffed = cosSimExpTens(pD, wD, pDT_again, wDT_again, ...
                           sigma, r, gD, isRel, isPer, periods, ...
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
dens_orig = buildExpTens(pAttr, w, sigma, r, groups, ...
                         isRel, isPer, periods, 'verbose', false);
dens_T    = buildExpTens(pT,    w, sigma, r, groups, ...
                         isRel, isPer, periods, 'verbose', false);
dens_D    = buildExpTens(pD,   wD, sigma, r, gD, ...
                         isRel, isPer, periods, 'verbose', false);
[pDT_4, wDT_4, gDT_4] = differenceEvents(pT, w, groups, {1, 0});
dens_DT   = buildExpTens(pDT_4, wDT_4, sigma, r, gDT_4, ...
                         isRel, isPer, periods, 'verbose', false);

% --- 10a. entropyExpTens on the struct; same answer as 9a. ---
H_orig_dens = entropyExpTens(dens_orig, 'method', 'renyi2', ...
                             'normalize', false, 'verbose', false);
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
