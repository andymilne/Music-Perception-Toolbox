%% diag_orbit_shapes.m
%  Localises the "reshape(vals, 1, nQ)" failure in evalExpTens's Möbius
%  path by calling the two orbit evaluators DIRECTLY (bypassing the
%  caller's reshape) and printing the returned size for every cell. The
%  cell whose ret_size has a element count != nQ is the culprit; its
%  shape tells us which branch collapses or broadcasts wrongly.
%
%  Run once with the toolbox on the path; send the whole table back:
%
%      >> diag_orbit_shapes
%
%  ret_size is the raw size() of what the evaluator returned. status is
%  OK when numel == nQ, BAD (with the numbers) when not, or EXC with the
%  error text if the evaluator itself threw.

rng(0, 'twister');
sigma = 15.0;

fprintf('\n%-4s %-4s %-3s %-4s %-5s %-14s %s\n', ...
    'rel', 'per', 'r', 'K', 'nQ', 'ret_size', 'status');
fprintf('%s\n', repmat('-', 1, 64));

relPer = {[false false], [true false], [false true], [true true]};

for rpi = 1:numel(relPer)
    v = relPer{rpi};
    isRel = v(1); isPer = v(2);
    P = 1200 * double(isPer);
    span = 1200 * double(isPer) + 3600 * double(~isPer);
    for r = [2 3 4]
        for K = [6]
            p = span * rand(K, 1);
            w = 0.2 + 0.8 * rand(K, 1);
            for nQ = [1 5]
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
                fprintf('%-4d %-4d %-3d %-4d %-5d %-14s %s\n', ...
                    isRel, isPer, r, K, nQ, retSize, status);
            end
        end
    end
end
fprintf('\n');
