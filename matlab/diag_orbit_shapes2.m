%% diag_orbit_shapes2.m
%  Wider version of diag_orbit_shapes: the K=6 grid was all OK, so the
%  failing cell is at larger K or larger nQ (nQ=200 is where the relative
%  path switches to its factored strategy, which the earlier probe never
%  exercised). This sweeps the benchmark's actual grid --- K in
%  {6,12,24,48}, nQ in {1,200} --- calling the two evaluators DIRECTLY and
%  printing the returned size, so the BAD row pinpoints the branch.
%
%      >> diag_orbit_shapes2
%
%  Send the whole table (or just the BAD/EXC rows, plus a couple of OK
%  rows around them for context).

rng(0, 'twister');
sigma = 15.0;

fprintf('\n%-4s %-4s %-3s %-4s %-5s %-16s %s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'ret_size', 'status');
fprintf('%s\n', repmat('-', 1, 70));

relPer = {[false false], [true false], [false true], [true true]};

for rpi = 1:numel(relPer)
    v = relPer{rpi};
    isRel = v(1); isPer = v(2);
    P = 1200 * double(isPer);
    span = 1200 * double(isPer) + 3600 * double(~isPer);
    for r = [2 3 4]
        for K = [6 12 24 48]
            p = span * rand(K, 1);
            w = 0.2 + 0.8 * rand(K, 1);
            for nQ = [1 200]
                if isRel
                    X = span * rand(r - 1, nQ);
                else
                    X = span * rand(r, nQ);
                end
                retSize = '-';
                status = '';
                try
                    if isRel
                        vals = mobius.evalOrbitRel(p(:), w(:), sigma, r, X, ...
                            'is_per', isPer, 'period', P);
                    else
                        vals = mobius.evalOrbitAbs(p(:), w(:), sigma, r, X, ...
                            'is_per', isPer, 'period', P);
                    end
                    retSize = mat2str(size(vals));
                    if numel(vals) == nQ
                        status = 'OK';
                    else
                        status = sprintf('BAD numel=%d expected=%d', ...
                            numel(vals), nQ);
                    end
                catch e
                    status = ['EXC ' e.message];
                end
                fprintf('%-4d %-4d %-3d %-4d %-5d %-16s %s\n', ...
                    isRel, isPer, r, K, nQ, retSize, status);
            end
        end
    end
end
fprintf('\n');
