%% test_renyi2_truncation.m — per-call kernel controls reach Rényi-2
%
%  entropyMaet(..., 'method', 'renyi2') forms H_2 = -log_b(<T,T> / Z^2)
%  with <T,T> = simMaet(T, T, 'normalize', 'none'). The total mass Z is
%  closed-form and independent of the kernel cutoff, so a change of
%  'truncationSigmas' must move H_2 by exactly
%  -log_b(<T,T>_k / <T,T>_default), and the kernel-covariance
%  change-of-variables term log det(Sigma) / 2 must be added once at
%  every width. Mirrors python/tests/test_renyi2_truncation.py.
%
%  Standalone-runnable. When invoked from test_mpt.m the existing
%  `results` cell is appended to; when run alone, results print on the
%  fly and a summary follows.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

h2 = @(d, varargin) entropyMaet(d, 'method', 'renyi2', ...
                                'verbose', false, varargin{:});
ipSelf = @(d, varargin) simMaet(d, d, 'normalize', 'none', ...
                                'verbose', false, varargin{:});

%% ---- Periodic relative r = 2 density ----

densPer = buildMaet([0 400 700 1000], ones(1, 4), 60, 2, true, true, ...
                    1200, 'verbose', false);

h6 = h2(densPer, 'truncationSigmas', 6);
h3 = h2(densPer, 'truncationSigmas', 3);
results{end+1,1} = 'renyi2.truncation: 3 sigma moves the value by a truncation-sized amount';
results{end,2}   = (h3 ~= h6) && abs(h3 - h6) < 1e-3;

% Z^2 from the accuracy-floor value; Z does not depend on the cutoff.
hInf  = h2(densPer, 'truncationSigmas', Inf, 'base', 2);
ipInf = ipSelf(densPer, 'truncationSigmas', Inf);
z2    = ipInf * 2^hInf;
ip3   = ipSelf(densPer, 'truncationSigmas', 3);
want  = -log2(ip3 / z2);
results{end+1,1} = 'renyi2.truncation: equals -log2(simMaet(T,T,none,3) / Z^2)';
results{end,2}   = abs(h3 - want) <= 1e-12 * abs(want);

% [] leaves the global default in force.
prevTs = mptDefaults('truncationSigmas');
mptDefaults('truncationSigmas', 6);
hOmit  = h2(densPer);
hEmpty = h2(densPer, 'truncationSigmas', []);
mptDefaults('truncationSigmas', Inf);
hGlobInf = h2(densPer);
mptDefaults('truncationSigmas', prevTs);
results{end+1,1} = 'renyi2.truncation: [] equals the global default';
results{end,2}   = (hOmit == hEmpty) && (hOmit == h6);
results{end+1,1} = 'renyi2.truncation: Inf per call equals the Inf global default';
results{end,2}   = abs(hInf - hGlobInf) <= 1e-14 * abs(hInf) ...
                   && abs(hInf - h6) <= 1e-7 * abs(hInf);

%% ---- Kernel covariance: the log det term is added once ----

Sigma = [1.0 0.3; 0.3 0.8];
Pk = [0.0 2.2 4.1 7.0; 0.0 1.9 4.4 6.3];
densK = buildMaet({Pk}, {ones(size(Pk))}, {Sigma}, 2, false, false, 0, ...
                  false, 'verbose', false);
hK6 = h2(densK, 'truncationSigmas', 6, 'base', exp(1));
hK3 = h2(densK, 'truncationSigmas', 3, 'base', exp(1));
ipK6 = ipSelf(densK, 'truncationSigmas', 6);
ipK3 = ipSelf(densK, 'truncationSigmas', 3);
dWant = -log(ipK3 / ipK6);
results{end+1,1} = 'renyi2.truncation: kernel covariance, shift equals -log(<T,T>_3/<T,T>_6)';
results{end,2}   = (hK3 ~= hK6) && abs((hK3 - hK6) - dWant) <= 1e-10 * abs(dWant) + 1e-15;

Ps = [0.0 1.5 3.1 4.0; 0.0 1.2 2.9 4.6];
s = 0.8;
densM = buildMaet({Ps}, {ones(size(Ps))}, {s^2 * eye(2)}, 2, false, ...
                  false, 0, false, 'verbose', false);
densS = buildMaet({Ps}, {ones(size(Ps))}, {s}, 2, false, false, 0, ...
                  false, 'verbose', false);
hM = h2(densM, 'truncationSigmas', 3);
hS = h2(densS, 'truncationSigmas', 3);
results{end+1,1} = 'renyi2.truncation: sigma^2 I matrix equals scalar sigma at 3 sigma';
results{end,2}   = abs(hM - hS) <= 1e-12 * abs(hS);

%% ---- Sub-density branch (relative r = 1 attribute) ----

Tm = [0.0 0.3 0.9 1.4];
wsOld = warning('off', 'all');
densSub = buildMaet({Ps, Tm}, {ones(size(Ps)), ones(size(Tm))}, ...
                    {0.8, 0.2}, [2 1], [false true], [false false], ...
                    [0 0], [false true], 'verbose', false);
warning(wsOld);
results{end+1,1} = 'renyi2.truncation: width reaches the sub-density branch';
results{end,2}   = h2(densSub, 'truncationSigmas', 3) ~= ...
                   h2(densSub, 'truncationSigmas', 6);

%% ---- kernelPrecision is accepted ----

hSingle = h2(densPer, 'truncationSigmas', 6, 'kernelPrecision', 'single');
results{end+1,1} = 'renyi2.truncation: kernelPrecision single accepted';
results{end,2}   = abs(hSingle - h6) <= 1e-5 * abs(h6);

%% ---- Standalone summary ----

if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:,2)));
    nFail = numel(results(:,1)) - nPass;
    fprintf('\n=== test_renyi2_truncation: %d passed, %d failed (of %d) ===\n', ...
        nPass, nFail, numel(results(:,1)));
    clear cleanupDefaults
    if nFail > 0
        for ii = 1:size(results, 1)
            if ~isequal(results{ii, 2}, true)
                fprintf('  FAIL  %s\n', results{ii, 1});
            end
        end
    end
end
