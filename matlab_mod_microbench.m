% Profile cost of mod-based vs floor-based periodic wrap in MATLAB
% on a chunk size representative of demo_expTensorPlots periodic
% configs (dim=3, nJ=343, nQ=20000). Run from the repo root.

dim  = 3;  nJ = 343;  nQ = 20000;
sigma  = 10;  period = 1200;
rng(0);
C = rand(dim, nJ) * 1200;
X = rand(dim, nQ) * 1200;
D = reshape(C, [dim, nJ, 1]) - reshape(X, [dim, 1, nQ]);

n_runs = 5;

% Method 1: mod-based wrap (current MATLAB code)
fprintf('Method 1: mod(D + period/2, period) - period/2\n');
times = zeros(n_runs, 1);
mod(D + period/2, period) - period/2;   % warm
for i = 1:n_runs
    t = tic;
    W1 = mod(D + period/2, period) - period/2;
    times(i) = toc(t);
end
fprintf('  median: %.1f ms\n', median(times)*1000);

% Method 2: floor-based wrap (mathematically equivalent everywhere)
fprintf('Method 2: D - period .* floor(D / period + 0.5)\n');
times = zeros(n_runs, 1);
D - period .* floor(D / period + 0.5);  % warm
for i = 1:n_runs
    t = tic;
    W2 = D - period .* floor(D / period + 0.5);
    times(i) = toc(t);
end
fprintf('  median: %.1f ms\n', median(times)*1000);

% Verify equivalence
maxd = max(abs(W1(:) - W2(:)));
fprintf('  max |method1 - method2| = %.3e\n', maxd);
