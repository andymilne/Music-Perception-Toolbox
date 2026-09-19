function v = preMaetCovParams(C)
%PREMAETCOVPARAMS  The generating flag and scalars of a covariance, or [].
%
%   Returns [differenced sdValue sdInterval sdShift]. Recovered from
%   the structure rather than remembered: each case of kernelCov is a
%   non-negative combination of three fixed basis matrices, so the
%   squared widths are fitted by least squares for each case in turn
%   and the result verified by rebuilding the matrix. A covariance of
%   neither family returns [] and cannot be written to CSV.
%
%   See also WRITEPREMAET, KERNELCOV.
v = [];
C = double(C);
r = size(C, 1);
if r < 2 || size(C, 1) ~= size(C, 2)
    return;
end
J = ones(r);
ddt = 2 * eye(r) - diag(ones(r - 1, 1), 1) - diag(ones(r - 1, 1), -1);
S = tril(ones(r, r - 1), -1);
P = eye(r) - J / r;
pssp = P * (S * S') * P;
cases = {true, {ddt, eye(r), J}; false, {eye(r), pssp, J}};
for c = 1:size(cases, 1)
    differenced = cases{c, 1};
    basis = cases{c, 2};
    A = [basis{1}(:), basis{2}(:), basis{3}(:)];
    sq = A \ C(:);
    if min(sq) < -1e-12
        continue;
    end
    vals = sqrt(max(sq(:)', 0));
    try
        rebuilt = kernelCov(r, 'differenced', differenced, ...
            'sdValue', vals(1), 'sdInterval', vals(2), ...
            'sdShift', vals(3));
    catch
        continue;
    end
    if max(max(abs(rebuilt - C))) <= 1e-9
        v = [double(differenced), vals];
        return;
    end
end
end
