%% test_rel_per_full_image.m
%  Tests for the full-image relative-periodic inner product. Twin of the
%  Python tests/test_rel_per_full_image.py.
%
%  The relative-periodic inner product marginalises a rigid common
%  shift. Taking that average over a kernel carrying every periodic
%  image yields the lattice-sum (full-image) measure exactly; taking it
%  over a nearest-image kernel yields a third measure, which departs
%  from both the full-image and single-image forms as sigma/period
%  grows.
%
%  Cross-language agreement is checked against
%  rel_per_full_image_parity.json, whose values were verified on the
%  Python side against an independent rank-(r-1) image-lattice sum to
%  1.6e-13. Recomputing that lattice sum here would duplicate a
%  fiddly reference rather than test the port, so the stored fixture is
%  used instead -- the same convention as ma_dispatch_parity.json.
%
%  No local functions: this file is executed as a script from
%  test_mpt.m.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
else
    standalone = false;
end

rpP = 1200;

% --- image count: zero where musical work sits, positive where the
%     measures diverge, monotone in sigma/period ---------------------
rpSops   = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50];
rpCounts = zeros(1, numel(rpSops));
for ii = 1:numel(rpSops)
    rpCounts(ii) = internal.relPerImageCount(rpSops(ii) * rpP, rpP, 6);
end
results(end+1, :) = {'relPer imageCount: zero at sigma/P <= 0.05', ...
    all(rpCounts(1:2) == 0)}; %#ok<*SAGROW>
results(end+1, :) = {'relPer imageCount: positive at sigma/P >= 0.10', ...
    all(rpCounts(3:end) >= 1)};
results(end+1, :) = {'relPer imageCount: monotone in sigma/P', ...
    issorted(rpCounts)};
results(end+1, :) = {'relPer imageCount: tighter floor needs >= images', ...
    internal.relPerImageCount(0.20 * rpP, rpP, Inf) >= ...
    internal.relPerImageCount(0.20 * rpP, rpP, 4)};

% Parity of the count itself with Python (None/6 -> [0 0 1 2 3],
% Inf -> [0 1 1 2 3] at sigma/P = 0.0125, 0.05, 0.10, 0.20, 0.30).
rpParitySops = [0.0125, 0.05, 0.10, 0.20, 0.30];
rpDefaultCnt = arrayfun(@(s) internal.relPerImageCount(s * rpP, rpP, 6), ...
                        rpParitySops);
rpExactCnt   = arrayfun(@(s) internal.relPerImageCount(s * rpP, rpP, Inf), ...
                        rpParitySops);
results(end+1, :) = {'relPer imageCount: default floor matches Python', ...
    isequal(rpDefaultCnt, [0 0 1 2 3])};
results(end+1, :) = {'relPer imageCount: accuracy floor matches Python', ...
    isequal(rpExactCnt, [0 1 1 2 3])};

% --- degenerate inputs return zero rather than raising ---------------
results(end+1, :) = {'relPer imageCount: degenerate inputs -> 0', ...
    internal.relPerImageCount(NaN, rpP, 6) == 0 && ...
    internal.relPerImageCount(100, 0,   6) == 0 && ...
    internal.relPerImageCount(100, -1,  6) == 0 && ...
    internal.relPerImageCount(Inf, rpP, 6) == 0};

% --- cross-language parity of the measure itself ---------------------
rpJson = fullfile(fileparts(mfilename('fullpath')), ...
                  'rel_per_full_image_parity.json');
if exist(rpJson, 'file') ~= 2
    results(end+1, :) = {'relPer full-image: parity fixture present', false};
else
    rpFix = jsondecode(fileread(rpJson));
    rpP1  = rpFix.p(:);
    rpP2  = rpFix.q(:);
    rpW1  = rpFix.w(:);
    rpW2  = rpFix.v(:);
    rpPer = rpFix.period;
    for ii = 1:numel(rpFix.cases)
        cse = rpFix.cases(ii);
        got = cosSimExpTens(rpP1, rpW1, rpP2, rpW2, cse.sigma, cse.r, ...
                            true, true, rpPer, ...
                            'method', 'mobius', 'verbose', false);
        results(end+1, :) = { ...
            sprintf('relPer full-image: r=%d sigma/P=%.2f matches Python', ...
                    cse.r, cse.sigmaOverP), ...
            abs(got - cse.cos) <= 1e-12};
    end
end


if standalone
    nPass = sum(cellfun(@(x) isequal(x, true), results(:, 2)));
    nFail = size(results, 1) - nPass;
    fprintf('test_rel_per_full_image: %d passed, %d failed\n', nPass, nFail);
    for ii = 1:size(results, 1)
        if ~isequal(results{ii, 2}, true)
            fprintf('  FAIL  %s\n', results{ii, 1});
        end
    end
end
