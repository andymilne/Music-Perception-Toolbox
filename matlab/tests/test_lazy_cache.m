%% test_lazy_cache.m — the joint fields of a lazy density are built once
%
%  A density is a struct, passed by value, so a consumer that builds the
%  joint (expensive) fields of a lazy density cannot leave them on the
%  caller's struct. buildMaet gives a lazy density a handle,
%  internal.MaetCache, in its field lazyCache, shared by every copy of the
%  struct: the first call that needs the fields
%  (internal.ensureMaetExpensive) stores the full density there, and later
%  calls, on the struct or any copy of it, take it from there, as Python
%  keeps the fields on the density object. Pinned here: that a consumer's
%  call on a copy fills the cache the caller's struct sees, and that a
%  later call takes what the cache holds; that it holds what a rebuild
%  returns; that a copy whose values, weights, or sigma have been edited
%  is rebuilt, and evaluates as a density built from the edited inputs,
%  and that the original then evaluates as before; that edited names and
%  wrap, and a kernel covariance, come back as the struct holds them;
%  that an eager density, a struct without the cache, and a single
%  multiset are handled; and that the cache is not saved with the
%  density.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults_lc
    cleanupDefaults_lc = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

% --- an onset at r = 1 beside three pitches at r = 2 ---
rng(21, 'twister');
lc_N = 40;
lc_on = sort(20 * rand(1, lc_N));
lc_pit = 60 + 12 * rand(3, lc_N);
lc_args = {{lc_on, lc_pit}, {ones(1, lc_N), ones(3, lc_N)}, [0.3 0.5], ...
           [1 2], [false false], [false false], [0 0]};
lc_d = buildMaet(lc_args{:}, 'verbose', false);
lc_X = [20 * rand(1, 60); 60 + 12 * rand(2, 60)];
lc_ev = @(d) evalMaet(d, lc_X, 'method', 'centres', 'truncationSigmas', 6, ...
                      'verbose', false);

results{end + 1, 1} = 'lazy_cache: a lazy density carries an empty cache';
results{end, 2} = isfield(lc_d, 'lazyCache') ...
    && isa(lc_d.lazyCache, 'internal.MaetCache') ...
    && ~lc_d.lazyCache.filled && ~isfield(lc_d, 'Centres');

% A consumer is handed a copy of the struct; its build fills the cache
% that the caller's struct sees.
lc_v = lc_ev(lc_d);
results{end + 1, 1} = 'lazy_cache: a consumer''s call on a copy fills the caller''s cache';
results{end, 2} = lc_d.lazyCache.filled ...
    && isfield(lc_d.lazyCache.dens, 'Centres') && any(lc_v > 0);

% What the cache holds is what a rebuild returns.
lc_full = internal.ensureMaetExpensive(lc_d);
lc_rebuilt = internal.ensureMaetExpensive(rmfield(lc_d, 'lazyCache'));
results{end + 1, 1} = 'lazy_cache: the cache holds what a rebuild returns';
results{end, 2} = isequal(lc_full, lc_rebuilt);

% A later call takes the build from the cache rather than building again:
% a mark put on the cached density comes back with it.
lc_d.lazyCache.dens.lcMark = true;
lc_marked = internal.ensureMaetExpensive(lc_d);
lc_d.lazyCache.dens = rmfield(lc_d.lazyCache.dens, 'lcMark');
results{end + 1, 1} = 'lazy_cache: a later call returns the cached build';
results{end, 2} = isfield(lc_marked, 'lcMark') && isequal(lc_ev(lc_d), lc_v);

% A copy whose fields have been edited shares the cache but is rebuilt:
% it evaluates as a density built from the edited inputs. The last case
% returns to the original inputs after the cache was replaced.
lc_names = {'values', 'weights', 'sigma', 'the original inputs'};
for lc_e = 1:4
    lc_c = lc_d;
    lc_a = lc_args;
    switch lc_e
        case 1
            lc_c.pAttr{2}(1, :) = lc_c.pAttr{2}(1, :) + 0.25;
            lc_a{1}{2}(1, :) = lc_a{1}{2}(1, :) + 0.25;
        case 2
            lc_c.w{1}(1, 1:2:end) = 0.5;
            lc_a{2}{1}(1, 1:2:end) = 0.5;
        case 3
            lc_c.sigma = [0.6 0.4];
            lc_a{3} = [0.6 0.4];
    end
    lc_ref = lc_ev(buildMaet(lc_a{:}, 'lazy', false, 'verbose', false));
    results{end + 1, 1} = sprintf(['lazy_cache: a copy with edited %s ' ...
        'evaluates as built from them'], lc_names{lc_e});
    results{end, 2} = lcClose(lc_ev(lc_c), lc_ref) ...
        && (lc_e < 4 || lcClose(lc_ref, lc_v));
end

% Edited names and wrap come back as the struct holds them.
lc_dp = buildMaet({lc_on, mod(lc_pit, 12)}, lc_args{2}, [0.3 0.3], [1 2], ...
    [false false], [false true], [0 12], 'verbose', false);
internal.ensureMaetExpensive(lc_dp);
lc_c = lc_dp;
lc_c.names = {'onset', 'pitch class'};
lc_c.wrap = {'full-image', 'single-image'};
lc_f = internal.ensureMaetExpensive(lc_c);
lc_fp = internal.ensureMaetExpensive(lc_dp);
results{end + 1, 1} = 'lazy_cache: edited names and wrap come back as edited';
results{end, 2} = isequal(lc_f.names, {'onset', 'pitch class'}) ...
    && isequal(lc_f.wrap, {'full-image', 'single-image'}) ...
    && isequal(lc_fp.wrap, lc_dp.wrap) && isequal(lc_fp.names, lc_dp.names);

% A kernel covariance comes back from the cache.
lc_A = randn(2, 2);
lc_Sig = lc_A * lc_A.' * 0.05 + 0.02 * eye(2);
lc_dk = buildMaet({10 * rand(2, lc_N), lc_on}, {ones(2, lc_N), ones(1, lc_N)}, ...
    {lc_Sig, 0.1}, [2 1], [false false], [false false], [0 0], ...
    [false true], 'verbose', false);
lc_k1 = internal.ensureMaetExpensive(lc_dk);
lc_k2 = internal.ensureMaetExpensive(lc_dk);
results{end + 1, 1} = 'lazy_cache: a kernel covariance comes back from the cache';
results{end, 2} = lc_dk.lazyCache.filled && isequal(lc_k1, lc_k2) ...
    && isequal(lc_k2.kernelCov, lc_dk.kernelCov) && isfield(lc_k2, 'Centres');

% An eager density has no cache and passes through unchanged.
lc_eager = buildMaet(lc_args{:}, 'lazy', false, 'verbose', false);
results{end + 1, 1} = 'lazy_cache: an eager density has no cache and passes through';
results{end, 2} = ~isfield(lc_eager, 'lazyCache') ...
    && isequal(internal.ensureMaetExpensive(lc_eager), lc_eager);

% A single multiset fills its cache on the single-multiset centres path.
lc_s = buildMaet(60 + 12 * rand(1, 8), ones(1, 8), 0.5, 2, false, false, 0, ...
                 'verbose', false);
lc_vs = evalMaet(lc_s, 60 + 12 * rand(2, 20), 'method', 'centres', ...
                 'verbose', false);
results{end + 1, 1} = 'lazy_cache: a single multiset fills its cache';
results{end, 2} = lc_s.lazyCache.filled && any(lc_vs > 0);

% The cache is not saved with the density: a loaded density builds its
% fields again on first use, to the same values.
lc_file = [tempname '.mat'];
save(lc_file, 'lc_d');
lc_loaded = load(lc_file);
delete(lc_file);
results{end + 1, 1} = 'lazy_cache: the cache is not saved with the density';
results{end, 2} = ~lc_loaded.lc_d.lazyCache.filled ...
    && isequal(lc_ev(lc_loaded.lc_d), lc_v) && lc_loaded.lc_d.lazyCache.filled;

clear lc_N lc_on lc_pit lc_args lc_d lc_X lc_ev lc_v lc_full lc_rebuilt ...
      lc_marked lc_names lc_e lc_c lc_a lc_ref lc_dp lc_f lc_fp lc_A lc_Sig ...
      lc_dk lc_k1 lc_k2 lc_eager lc_s lc_vs lc_file lc_loaded

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for ii = 1:size(results, 1)
        if results{ii, 2}
            fprintf('  PASS  %s\n', results{ii, 1});
        else
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
    fprintf('\n=== test_lazy_cache: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults_lc
    if nFail > 0
        error('test_lazy_cache:failed', '%d test(s) failed.', nFail);
    end
end


function ok = lcClose(a, b)
    % Agreement to rounding, relative to the largest value, which must be
    % positive for the comparison to mean anything.
    scale = max(max(abs(b)), 1e-300);
    ok = isequal(size(a), size(b)) && max(abs(a - b)) <= 1e-12 * scale ...
         && any(b > 0);
end
