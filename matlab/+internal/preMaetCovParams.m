function v = preMaetCovParams(C)
%PREMAETCOVPARAMS  The three generating scalars of a covariance, or [].
%
%   Recovered from the structure rather than remembered: with D the
%   difference operator,
%
%       Sigma = sdPosition^2 D D' + sdInterval^2 I + sdShift^2 J
%
%   so the diagonal is 2a + b + c, the first off-diagonal c - a, and every
%   entry further out c. Three entries determine the family, and the
%   result is verified by rebuilding the matrix: a covariance of any other
%   shape returns [] and cannot be written to CSV. The recovery needs
%   r >= 3, there being no entry two off the diagonal at r = 2.
%
%   See also WRITEPREMAET, INTERVALKERNELCOV.
v = [];
C = double(C);
r = size(C, 1);
if r < 3 || size(C, 1) ~= size(C, 2)
    return;
end
c = C(1, 3);
a = c - C(1, 2);
b = C(1, 1) - 2 * a - c;
if min([a b c]) < -1e-12
    return;
end
vals = sqrt(max([a b c], 0));
rebuilt = intervalKernelCov(r, 'sdPosition', vals(1), ...
    'sdInterval', vals(2), 'sdShift', vals(3));
if max(max(abs(rebuilt - C))) > 1e-9
    return;
end
v = vals;
end
