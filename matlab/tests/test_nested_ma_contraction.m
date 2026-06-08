%% test_nested_ma_contraction.m — multi-attribute nested contraction:
%  a nested attribute tensored with one or more plain attributes.
%
%  Mirror of the Python test_nested_ma_contraction.py. The fast tree
%  contraction previously handled only a single-attribute density; tensoring
%  any further attribute (e.g. an inversion flag) made it decline, and the
%  multi-attribute case routed to the joint-tuple enumeration --- which both
%  blows up combinatorially at inner r >= 2 and (in the MA tensor build)
%  mis-shapes a nested attribute's per-event tuples. The MA path now routes
%  each nested factor through the contraction and each plain factor through
%  mobius.maPerAttrInnerMatrix, combining them per event-pair (JMM Eq 3.4):
%      <X,Y> = sum_{i,j} prod_a I_a(i,j).
%  Per-attribute prefactors are constant and cancel in the cosine, so mixing
%  the two matrix conventions is exact.
%
%  Reference values are the Python method='contract' results for identical
%  constructions, themselves verified equal to the exact Python Bulger
%  enumeration. (Bulger is not used as the MATLAB reference here: its MA
%  tensor build mis-shapes nested tuples, so method='bulger' on a nested
%  multi-attribute density is unsupported.)

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

GTOL = 1e-5;     % cross-language vs the Python contract goldens

C = 0; E = 4; G = 7; Eb = 3;
IVI  = {[C E G], [G 11 2], [C E G]};
ivi  = {[C Eb G], [G 11 2], [C Eb G]};
IVI2 = {[C E G C E G], [G 11 2 G 11 2], [C E G C E G]};
mel  = {[2 5 9], [9 1 4], [2 5 9]};

% --- single-event: flags match (-> harmonic match) and differ (-> ~0) ---
for ri = [1 2]
    X = madens({IVI}, 0.5, ri);
    cMatch = cosSimExpTens(X, madens({IVI}, 0.5, ri), 'method', 'contract', 'verbose', false);
    cDiff  = cosSimExpTens(X, madens({ivi}, -0.5, ri), 'method', 'contract', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: 1ev flags match ri=%d (=1)', ri); %#ok<*SAGROW>
    results{end, 2}   = abs(cMatch - 1.0) < GTOL;
    results{end+1, 1} = sprintf('nested-ma: 1ev flags differ ri=%d (~0)', ri);
    results{end, 2}   = abs(cDiff) < 1e-4;
end

% --- single-event, unequal nested cardinality (3 vs 6) plus the flag ---
ref_uneq = [1.0, 0.837924];
for k = 1:2
    ri = k;
    X = madens({IVI}, 0.5, ri);
    Y = madens({IVI2}, 0.5, ri);
    c  = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
    cR = cosSimExpTens(Y, X, 'method', 'contract', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: 1ev UNEQ 3v6 ri=%d == Python golden', ri);
    results{end, 2}   = abs(c - ref_uneq(k)) < GTOL && abs(c - cR) < 1e-9;
end

% --- multi-event (2 events per side): per-event-pair matrix combination ---
ref_multi = [0.645572, 0.527059];
for k = 1:2
    ri = k;
    X = madens({IVI, ivi}, [0.5 -0.5], ri);
    Y = madens({IVI, mel}, [0.5 0.5], ri);
    c    = cosSimExpTens(X, Y, 'method', 'contract', 'verbose', false);
    cSelf = cosSimExpTens(X, X, 'method', 'contract', 'verbose', false);
    results{end+1, 1} = sprintf('nested-ma: 2ev multi-event ri=%d == Python golden, self=1', ri);
    results{end, 2}   = abs(c - ref_multi(k)) < GTOL && abs(cSelf - 1.0) < GTOL;
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
function d = madens(events, flags, rIn)
    % Two-attribute density: a nested harmonic attribute (outer relative,
    % periodic) tensored with a 1-D non-periodic flag attribute. events is a
    % cell over events, each a cell of chords (pitch-class row vectors).
    SIG = 0.15; SF = 0.1; P = 12.0;
    nCh = numel(events{1});
    nSlot = numel(events{1}{1});
    tags = [];
    for k = 1:nCh; tags = [tags, (k - 1) * ones(1, nSlot)]; end
    N = numel(events);
    p0 = zeros(nCh * nSlot, N);
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
