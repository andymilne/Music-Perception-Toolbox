%% test_nested_ma_contraction.m — multi-attribute nested contraction, and
%  the nested-tuple enumeration used by the Bulger path.
%
%  Mirror of the Python test_nested_ma_contraction.py. Two things are guarded:
%
%  1. MULTI-ATTRIBUTE ROUTING. The fast tree contraction previously handled
%     only a single-attribute density; tensoring any further attribute (e.g.
%     an inversion flag) made it decline. The MA path now routes each nested
%     factor through the contraction and each plain factor through
%     mobius.maPerAttrInnerMatrix, combining them per event-pair (JMM Eq 3.4):
%         <X,Y> = sum_{i,j} prod_a I_a(i,j).
%     Per-attribute prefactors are constant and cancel in the cosine, so
%     mixing the two matrix conventions is exact.
%
%  2. NESTED-TUPLE ENUMERATION (the Bulger / eager build). The leaf
%     enumeration mapped combination positions to slot indices with
%     rowset(nchoosek(1:k, r0)); at r0 = 1 the (k x 1) position column
%     followed the row vector rowset's orientation, collapsing k single-slot
%     combinations into one k-slot combination. Any nesting level with r = 1
%     and k > 1 (e.g. one note read per chord) was mis-enumerated --- a crash
%     in the MA tensor build, wrong tuples in the single-attribute build.
%
%  Reference values are the Python results for identical constructions; the
%  contraction and the (now-correct) Bulger enumeration must agree with them
%  and with each other.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

ATOL = 1e-9;     % contract vs bulger within MATLAB
GTOL = 1e-5;     % cross-language vs the Python goldens

C = 0; E = 4; G = 7; Eb = 3;
IVI  = {[C E G], [G 11 2], [C E G]};
ivi  = {[C Eb G], [G 11 2], [C Eb G]};
IVI2 = {[C E G C E G], [G 11 2 G 11 2], [C E G C E G]};
mel  = {[2 5 9], [9 1 4], [2 5 9]};

% --- single-attribute nested: contract == bulger == Python, both arities.
%     The r=1 case is the direct guard for the leaf enumeration fix. ---
ref_sa = containers.Map({1, 2}, {0.667053, 0.111164});
Xsa = {[C E G], [G 11 2], [C E G]};
Ysa = {[C Eb G], [G 11 2], [C Eb G]};
for ri = [1 2]
    X = sadens(Xsa, ri);
    Y = sadens(Ysa, ri);
    cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
    cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: SA ri=%d contract==bulger==Python', ri); %#ok<*SAGROW>
    results{end, 2}   = abs(cC - cB) < ATOL && abs(cC - ref_sa(ri)) < GTOL;
end

% --- MA (nested + flag): flags match (-> harmonic match) and differ (~0) ---
for ri = [1 2]
    X = madens({IVI}, 0.5, ri);
    [cMc, cMb] = bothMethods(X, madens({IVI}, 0.5, ri));
    [cDc, cDb] = bothMethods(X, madens({ivi}, -0.5, ri));
    results{end+1, 1} = sprintf('nested-ma: 1ev flags match ri=%d (=1)', ri);
    results{end, 2}   = abs(cMc - 1) < GTOL && abs(cMb - 1) < GTOL && abs(cMc - cMb) < ATOL;
    results{end+1, 1} = sprintf('nested-ma: 1ev flags differ ri=%d (~0)', ri);
    results{end, 2}   = abs(cDc) < 1e-4 && abs(cDb) < 1e-4 && abs(cDc - cDb) < ATOL;
end

% --- MA, unequal nested cardinality (3 vs 6) plus the flag ---
ref_uneq = containers.Map({1, 2}, {1.0, 0.837924});
for ri = [1 2]
    [cC, cB] = bothMethods(madens({IVI}, 0.5, ri), madens({IVI2}, 0.5, ri));
    results{end+1, 1} = sprintf('nested-ma: 1ev UNEQ 3v6 ri=%d', ri);
    results{end, 2}   = abs(cC - ref_uneq(ri)) < GTOL && abs(cB - ref_uneq(ri)) < GTOL ...
                     && abs(cC - cB) < ATOL;
end

% --- MA, multi-event (2 events per side): per-event-pair combination ---
ref_multi = containers.Map({1, 2}, {0.645572, 0.527059});
for ri = [1 2]
    X = madens({IVI, ivi}, [0.5 -0.5], ri);
    Y = madens({IVI, mel}, [0.5 0.5], ri);
    [cC, cB] = bothMethods(X, Y);
    cSelf = cosSimExpTens(X, X, 'method', 'contract', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: 2ev multi-event ri=%d, self=1', ri);
    results{end, 2}   = abs(cC - ref_multi(ri)) < GTOL && abs(cB - ref_multi(ri)) < GTOL ...
                     && abs(cC - cB) < ATOL && abs(cSelf - 1) < GTOL;
end

% --- one-sided normalisation must route through the contraction too ---
%     Regression: before the fix, method='contract' + 'oneSidedDenom' errored
%     and 'auto' fell back to the joint-tuple enumeration. The contraction
%     returns the bare (xy, xx, yy) triple and the denominator is chosen
%     downstream, so it is correct for either normalisation. Doubled chords
%     (IVI2) against the single voicing (IVI), flags matching, give a clean
%     magnitude golden (Python: 8 at rIn=1, 64 at rIn=2; SA and MA alike).
ref_os = containers.Map({1, 2}, {8.0, 64.0});
for ri = [1 2]
    g = ref_os(ri); aTol = ATOL * max(1, g); gTol = GTOL * max(1, g);
    % single nested attribute (internal.nestedContract)
    Xs = sadens(IVI2, ri); Ys = sadens(IVI, ri);
    osC = cosSimExpTens(Xs, Ys, 'method', 'contract', 'normalize', 'oneSidedDenom', 'verbose', false);
    osB = cosSimExpTens(Xs, Ys, 'method', 'bulger',   'normalize', 'oneSidedDenom', 'verbose', false);
    osA = cosSimExpTens(Xs, Ys, 'normalize', 'oneSidedDenom', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: SA one-sided ri=%d contract==bulger==auto==Python', ri);
    results{end, 2}   = abs(osC - osB) < aTol && abs(osA - osB) < aTol && abs(osB - g) < gTol;
    % nested (x) flag (internal.nestedContractMA)
    Xm = madens({IVI2}, 0.5, ri); Ym = madens({IVI}, 0.5, ri);
    omC = cosSimExpTens(Xm, Ym, 'method', 'contract', 'normalize', 'oneSidedDenom', 'verbose', false);
    omB = cosSimExpTens(Xm, Ym, 'method', 'bulger',   'normalize', 'oneSidedDenom', 'verbose', false);
    omA = cosSimExpTens(Xm, Ym, 'normalize', 'oneSidedDenom', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: MA one-sided ri=%d contract==bulger==auto==Python', ri);
    results{end, 2}   = abs(omC - omB) < aTol && abs(omA - omB) < aTol && abs(omB - g) < gTol;
end


if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_nested_ma_contraction: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_nested_ma_contraction:failed', '%d test(s) failed.', nFail);
    end
end


% ----------------------------------------------------------------------
function [cC, cB] = bothMethods(X, Y)
    cC = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
    cB = cosSimExpTens(X, Y, 'method', 'bulger',   'verbose', false);
end


function d = sadens(chords, rIn)
    % Single nested harmonic attribute (outer relative, periodic).
    SIG = 0.15; P = 12.0;
    nCh = numel(chords); nValue = numel(chords{1});
    tags = [];
    for k = 1:nCh; tags = [tags, (k - 1) * ones(1, nValue)]; end
    p0 = [];
    for c = 1:nCh; p0 = [p0, chords{c}]; end
    sp = struct('tags', tags, 'r', [rIn nCh], 'sym', [true false], 'rel', [0 1]);
    d = buildExpTens({p0(:)}, {[]}, 'specs', {sp}, 'sigma', SIG, ...
                     'isPer', true, 'period', P, 'verbose', false);
end


function d = madens(events, flags, rIn)
    % Nested harmonic attribute (outer relative, periodic) tensored with a
    % 1-D non-periodic flag attribute. events is a cell over events, each a
    % cell of chords (pitch-class row vectors).
    SIG = 0.15; SF = 0.1; P = 12.0;
    nCh = numel(events{1}); nValue = numel(events{1}{1});
    tags = [];
    for k = 1:nCh; tags = [tags, (k - 1) * ones(1, nValue)]; end
    N = numel(events);
    p0 = zeros(nCh * nValue, N);
    for e = 1:N
        col = [];
        for c = 1:nCh; col = [col, events{e}{c}]; end
        p0(:, e) = col(:);
    end
    p1 = reshape(flags, 1, []);
    sp0 = struct('tags', tags, 'r', [rIn nCh], 'sym', [true false], 'rel', [0 1]);
    sp1 = struct('r', 1, 'sym', false, 'rel', false);
    d = buildExpTens({p0, p1}, {[], []}, 'specs', {sp0, sp1}, ...
                     'sigma', [SIG SF], 'isPer', [true false], ...
                     'period', [P 1.0], 'verbose', false);
end
