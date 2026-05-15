%% test_tensor_harmonicity_orbit.m — v2.2 tensorHarmonicity rewrite
%
%  Tests for the v2.2 rewrite of tensorHarmonicity, which bypasses
%  buildExpTens entirely and routes through mobius.evalOrbitRel.
%  Covers:
%    - Numerical equivalence to a hand-rolled call to mobius.evalOrbitRel
%      (the rewrite must not introduce extra factors).
%    - Cardinality > 3 (4-pitch chord with auto duplicate=4): the
%      v2.0/v2.1 centres path could not handle this without millions of
%      r-tuples; the Möbius method makes it routine.
%    - Batched dedup: rows whose canonical (sorted, translation-removed)
%      pitch sequences coincide return the same harmonicity.
%    - Normalize='gaussian' and 'pdf' apply the same constants the
%      centres path used.
%    - Cache key includes sigma and dup (different params -> different
%      cached values).
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    % Defaults isolation when run standalone (when invoked from
    % test_mpt.m the outer wrapper has already isolated defaults).
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

%% ---- Numerical equivalence to direct Möbius call ----

p = [0, 400, 700];
sigma = 12;
spec = {'harmonic', 12, 'powerlaw', 1};

h_th = tensorHarmonicity(p, [], sigma, 'spectrum', spec, 'verbose', false);

% Hand-rolled equivalent: same template, same query, raw Möbius-method call.
dup = numel(p);
[tmpl_p, tmpl_w] = addSpectra(zeros(dup, 1), ones(dup, 1), spec{:});
p_sorted = sort(p);
intervals = (p_sorted(2:end) - p_sorted(1)).';
h_orbit = mobius.evalOrbitRel(tmpl_p(:), tmpl_w(:), sigma, ...
    numel(p), intervals, 'is_per', false, 'period', 0);
h_orbit = h_orbit(1);

results{end+1,1} = 'tensorHarmonicity v2.2: matches direct mobius.evalOrbitRel call (1e-12)';
results{end,2}   = abs(h_th - h_orbit) < 1e-12;

%% ---- Cardinality > 3 (the killer feature) ----

% 4-pitch chord with auto duplicate=4. With a 12-partial harmonic template
% this gives 48 source positions and r=4, so K!/(K-r)! = 48*47*46*45 = 4.7M
% ordered tuples; the centres path materialises an array of size
% (3, 4.7M) ~= 110 MB just for the centres. The Möbius method skips this.
p4 = [0, 400, 700, 1200];   % major triad with octave on top
h4 = tensorHarmonicity(p4, [], sigma, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2: 4-pitch chord (K=4) returns finite scalar';
results{end,2}   = isfinite(h4) && isscalar(h4) && h4 > 0;

% Same chord transposed up an octave: same canonical key -> same value.
p4_up = p4 + 1200;
h4_up = tensorHarmonicity(p4_up, [], sigma, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2: transposition invariance on K=4 chord (1e-12)';
results{end,2}   = abs(h4 - h4_up) < 1e-12;

%% ---- Batched dedup: canonical key collapses transpositions ----

% Three rows with different absolute pitches but identical canonical
% (relative) form. Should all give the same harmonicity.
P_dedup = [0,    400,  700;     %#ok<NASGU>  major triad
           500,  900,  1200;
           -100, 300,  600];
h_dedup = tensorHarmonicity(P_dedup, [], sigma, 'spectrum', spec, ...
    'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2 batched: transposition-equivalent rows agree (1e-12)';
results{end,2}   = isequal(size(h_dedup), [3, 1]) ...
                && abs(h_dedup(1) - h_dedup(2)) < 1e-12 ...
                && abs(h_dedup(1) - h_dedup(3)) < 1e-12;

% Cross-check: scalar single-row matches.
h_scalar = tensorHarmonicity([0, 400, 700], [], sigma, 'spectrum', spec, ...
    'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2 batched: dedup matches scalar (1e-12)';
results{end,2}   = abs(h_dedup(1) - h_scalar) < 1e-12;

%% ---- Cache key includes sigma and dup ----

% Different sigma must produce different cached values for the same
% chord (cache key disambiguation).
P_two = [0, 400, 700;
         0, 400, 700];
h_s12 = tensorHarmonicity(P_two, [], 12, 'spectrum', spec, 'verbose', false);
h_s20 = tensorHarmonicity(P_two, [], 20, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2 batched: sigma=12 vs sigma=20 disambiguate';
results{end,2}   = abs(h_s12(1) - h_s20(1)) > 1e-6;

% Same chord, different duplicate -> different result.
h_dup_auto = tensorHarmonicity(P_two, [], 12, 'spectrum', spec, ...
    'duplicate', 0, 'verbose', false);  % auto -> dup=3
h_dup_1    = tensorHarmonicity(P_two, [], 12, 'spectrum', spec, ...
    'duplicate', 1, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2 batched: duplicate=0 vs 1 disambiguate';
results{end,2}   = abs(h_dup_auto(1) - h_dup_1(1)) > 1e-6;

%% ---- Normalize='gaussian' and 'pdf' apply consistent constants ----

% normalize='gaussian' multiplies by (2*pi*sigma^2)^(-dim/2) * sqrt(detM)
% with detM = 1/r and dim = r-1. Verify by computing the constant manually.
r = 3;
dim = r - 1;
detM = 1/r;
gaussConst = (2*pi*sigma^2)^(-dim/2) * sqrt(detM);

h_none = tensorHarmonicity([0, 400, 700], [], sigma, 'spectrum', spec, ...
    'normalize', 'none', 'verbose', false);
h_gauss = tensorHarmonicity([0, 400, 700], [], sigma, 'spectrum', spec, ...
    'normalize', 'gaussian', 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2: gaussian normalize applies (2 pi sigma^2)^{-dim/2} sqrt(1/r)';
results{end,2}   = abs(h_gauss - h_none * gaussConst) < 1e-10 * abs(h_gauss);

% normalize='pdf' additionally divides by sum(tmpl_w).
[tmpl_p_test, tmpl_w_test] = addSpectra(zeros(3, 1), ones(3, 1), spec{:});
sumW = sum(tmpl_w_test);
h_pdf = tensorHarmonicity([0, 400, 700], [], sigma, 'spectrum', spec, ...
    'normalize', 'pdf', 'verbose', false);
results{end+1,1} = 'tensorHarmonicity v2.2: pdf = gaussian / sum(tmpl_w) (1e-10 rel)';
results{end,2}   = abs(h_pdf - h_gauss / sumW) < 1e-10 * abs(h_pdf);

%% ---- Verbose flag prints an eval message in scalar mode ----
% Post-Stage-2c, the wrapper no longer hard-codes 'mobius' since
% the centres-vs-Möbius choice is made by evalExpTens's dispatcher.
% The printed message is now 'tensorHarmonicity: eval at K = ...'.

outScalarVerb = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', true);']);
results{end+1,1} = 'tensorHarmonicity v2.2 scalar: verbose=true prints eval message';
results{end,2}   = ~isempty(strfind(outScalarVerb, 'tensorHarmonicity: eval'));

outScalarSilent = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', false);']);
results{end+1,1} = 'tensorHarmonicity v2.2 scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outScalarSilent));

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_tensor_harmonicity_orbit: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    % Restore caller's pre-test defaults eagerly
    % (fires the helper's onCleanup destructor before
    % the conditional error below halts execution).
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
