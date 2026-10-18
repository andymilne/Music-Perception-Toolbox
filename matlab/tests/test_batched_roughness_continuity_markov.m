%% test_batched_roughness_continuity_markov.m — batched roughness, continuity, markovS
%
%  Each batched form must agree, row by row, with the single form, and
%  must merge the rows its key says are equivalent: repeated rows (in
%  any order) for roughness, jointly transposed rows for continuity, and
%  rotations for markovS.
%
%  Mirror of Python tests/test_batched_roughness_continuity_markov.py.
%
%  Standalone-runnable; appends to `results` when called from test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    addpath(fileparts(fileparts(mfilename('fullpath'))));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end

brcmSpec = {'harmonic', 8, 'powerlaw', 1};
brcmHz   = @(c) 261.63 * 2 .^ (c / 1200);

%% ---- roughness ----

brcmF = [brcmHz([0 400 700 NaN])
         brcmHz([0 300 700 NaN])
         brcmHz([0 400 700 1000])
         brcmHz([700 0 400 NaN])          % row 1, reordered
         NaN(1, 4)];

brcmR = roughness(brcmF);
brcmOk = isequal(size(brcmR), [5 1]) && isnan(brcmR(5));
for brcmI = 1:4
    brcmRow = brcmF(brcmI, ~isnan(brcmF(brcmI, :)));
    brcmOk = brcmOk && abs(brcmR(brcmI) - roughness(brcmRow)) ...
                       <= 1e-12 * max(1, abs(brcmR(brcmI)));
end
results{end+1,1} = 'roughness: batched rows match the single form';
results{end,2}   = brcmOk;

brcmRS = roughness(brcmF, [], 'spectrum', brcmSpec);
brcmOk = true;
for brcmI = 1:4
    brcmRow = brcmF(brcmI, ~isnan(brcmF(brcmI, :)));
    brcmOk = brcmOk && abs(brcmRS(brcmI) ...
        - roughness(brcmRow, [], 'spectrum', brcmSpec)) ...
        <= 1e-12 * max(1, abs(brcmRS(brcmI)));
end
results{end+1,1} = 'roughness: batched rows match the single form with a spectrum';
results{end,2}   = brcmOk;

brcmF1 = brcmHz([0 400 700]);
[brcmP, brcmW] = addSpectra(1200 * log2(brcmF1(:)), [], brcmSpec{:});
brcmByHand = roughness(2 .^ (brcmP / 1200), brcmW);
results{end+1,1} = 'roughness: spectrum equals adding the partials by hand';
results{end,2}   = abs(roughness(brcmF1, [], 'spectrum', brcmSpec) ...
                       - brcmByHand) <= 1e-12 * max(1, abs(brcmByHand));

results{end+1,1} = 'roughness: a reordered row equals the original';
results{end,2}   = abs(brcmRS(4) - brcmRS(1)) <= 1e-12 * max(1, brcmRS(1));

brcmT = roughness([brcmHz([0 400 700]); brcmHz([500 900 1200])], [], ...
                  'spectrum', brcmSpec);
results{end+1,1} = 'roughness: transpositions are not merged';
results{end,2}   = abs(brcmT(1) - brcmT(2)) > 1e-6;

brcmF2 = [brcmHz([0 400 700]); brcmHz([0 300 700])];
brcmWv = [1 0.5 0.8];
brcmRb = roughness(brcmF2, brcmWv);
brcmRm = roughness(brcmF2, repmat(brcmWv, 2, 1));
results{end+1,1} = 'roughness: broadcast and matrix weights agree';
results{end,2}   = max(abs(brcmRb - brcmRm)) <= 1e-12 * max(1, max(brcmRb)) ...
    && abs(brcmRb(1) - roughness(brcmF2(1, :), brcmWv)) ...
       <= 1e-12 * max(1, brcmRb(1));

brcmRa = roughness(brcmF2, [], 'spectrum', brcmSpec, 'average', true, ...
                   'pNorm', 2);
results{end+1,1} = 'roughness: average and pNorm pass through the batch';
results{end,2}   = abs(brcmRa(2) - roughness(brcmF2(2, :), [], ...
    'spectrum', brcmSpec, 'average', true, 'pNorm', 2)) ...
    <= 1e-12 * max(1, brcmRa(2));

results{end+1,1} = 'roughness: a non-positive frequency in a row errors';
results{end,2}   = throwsError(@() roughness([100 200; 0 300]));

results{end+1,1} = 'roughness: a weight vector of the wrong length errors';
results{end,2}   = throwsError(@() roughness(brcmF, ones(1, 3)));

%% ---- continuity ----

brcmSeq = [60 62 64 65 67
           60 64 62 60 NaN
           65 67 69 70 72              % row 1 up a fourth
           NaN NaN NaN NaN NaN] * 100;
brcmSig = 10;

brcmX = [69; 58; 74; 60] * 100;
[brcmC, brcmM] = continuity(brcmSeq, brcmX, brcmSig);
brcmOk = isequal(size(brcmC), [4 1]) && isequal(size(brcmM), [4 1]) ...
         && isnan(brcmC(4)) && isnan(brcmM(4));
for brcmI = 1:3
    brcmRow = brcmSeq(brcmI, ~isnan(brcmSeq(brcmI, :)));
    [brcmCs, brcmMs] = continuity(brcmRow, brcmX(brcmI), brcmSig);
    brcmOk = brcmOk && abs(brcmC(brcmI) - brcmCs) < 1e-12 ...
                    && abs(brcmM(brcmI) - brcmMs) < 1e-12;
end
results{end+1,1} = 'continuity: one query per row matches the single form';
results{end,2}   = brcmOk;

results{end+1,1} = 'continuity: jointly transposed rows agree';
results{end,2}   = abs(brcmC(3) - brcmC(1)) < 1e-12 ...
                   && abs(brcmM(3) - brcmM(1)) < 1e-12;

brcmXs = [5800 6900 7400];
[brcmC, brcmM] = continuity(brcmSeq(1:3, :), brcmXs, brcmSig, ...
                            'mode', 'lenient');
brcmOk = isequal(size(brcmC), [3 3]);
for brcmI = 1:3
    brcmRow = brcmSeq(brcmI, ~isnan(brcmSeq(brcmI, :)));
    [brcmCs, brcmMs] = continuity(brcmRow, brcmXs, brcmSig, ...
                                  'mode', 'lenient');
    brcmOk = brcmOk && max(abs(brcmC(brcmI, :) - brcmCs.')) < 1e-12 ...
                    && max(abs(brcmM(brcmI, :) - brcmMs.')) < 1e-12;
end
results{end+1,1} = 'continuity: shared queries match the single form';
results{end,2}   = brcmOk;

[brcmC, ~] = continuity(brcmSeq(1:3, :), repmat([6900 NaN], 3, 1), brcmSig);
results{end+1,1} = 'continuity: a NaN query gives NaN';
results{end,2}   = all(isnan(brcmC(:, 2))) && all(isfinite(brcmC(:, 1)));

brcmWc = [0.2 0.4 0.6 0.8 1.0];
[brcmC, ~]  = continuity(brcmSeq(1:3, :), [6900; 6900; 6900], brcmSig, ...
                         'w', brcmWc);
[brcmCs, ~] = continuity(brcmSeq(1, :), 6900, brcmSig, 'w', brcmWc);
[brcmC2, ~] = continuity(brcmSeq(1:3, :), [6900; 6900; 6900], brcmSig, ...
                         'w', repmat(brcmWc, 3, 1));
results{end+1,1} = 'continuity: shared and per-row weights';
results{end,2}   = abs(brcmC(1) - brcmCs) < 1e-12 ...
                   && max(abs(brcmC - brcmC2)) < 1e-12;

results{end+1,1} = 'continuity: x with the wrong number of rows errors';
results{end,2}   = throwsError(@() continuity(brcmSeq, zeros(2, 1), brcmSig));

[brcmC, brcmM] = continuity([3 5 7 7 9], 11, 0, 'mode', 'lenient');
results{end+1,1} = 'continuity: the vector form is unchanged';
results{end,2}   = isscalar(brcmC) && abs(brcmC - 3) < 1e-10 ...
                   && abs(brcmM - 6) < 1e-10;

%% ---- markovS ----

brcmPm = [0 3 6 10 12
          2 5 8 12 14                 % the clave rotated by 2
          0 4 8 12 NaN
          NaN NaN NaN NaN NaN];
brcmY = markovS(brcmPm, [], 16);
brcmOk = isequal(size(brcmY), [4 16]) && all(isnan(brcmY(4, :)));
for brcmI = 1:3
    brcmRow = brcmPm(brcmI, ~isnan(brcmPm(brcmI, :)));
    brcmOk = brcmOk && max(abs(brcmY(brcmI, :) ...
                               - markovS(brcmRow, [], 16))) < 1e-12;
end
results{end+1,1} = 'markovS: batched rows match the single form';
results{end,2}   = brcmOk;

results{end+1,1} = 'markovS: a rotated pattern gives the rotated prediction';
results{end,2}   = max(abs(brcmY(2, :) - circshift(brcmY(1, :), 2, 2))) < 1e-12;

brcmWm = [1 0.5 0.8 0.7 1];
brcmY = markovS(brcmPm(1:2, :), brcmWm, 16, 5);
results{end+1,1} = 'markovS: weights and S pass through the batch';
results{end,2}   = max(abs(brcmY(1, :) ...
                       - markovS([0 3 6 10 12], brcmWm, 16, 5))) < 1e-12 ...
    && max(abs(brcmY(2, :) - markovS(brcmPm(2, :), brcmWm, 16, 5))) < 1e-12;

brcmPs = [0 4 8 12; 1 5 9 13; 2 6 10 14];
brcmY = markovS(brcmPs, [], 16);
brcmOk = true;
for brcmI = 1:3
    brcmOk = brcmOk && max(abs(brcmY(brcmI, :) ...
                               - markovS(brcmPs(brcmI, :), [], 16))) < 1e-12;
end
results{end+1,1} = 'markovS: a symmetric pattern rotates back into place';
results{end,2}   = brcmOk;

results{end+1,1} = 'markovS: a non-integer position in a row errors';
results{end,2}   = throwsError(@() markovS([0 3; 1.5 4], [], 16));

results{end+1,1} = 'markovS: a position outside the period in a row errors';
results{end,2}   = throwsError(@() markovS([0 3; 1 16], [], 16));

clear -regexp ^brcm

%% ---- Standalone summary ----

if standalone
    nPass = 0; nFail = 0;
    for i = 1:size(results, 1)
        if results{i,2}
            nPass = nPass + 1;
            fprintf('  PASS  %s\n', results{i,1});
        else
            nFail = nFail + 1;
            fprintf('  FAIL  %s\n', results{i,1});
        end
    end
    fprintf(['\n=== test_batched_roughness_continuity_markov: %d passed, ' ...
             '%d failed (of %d) ===\n\n'], nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_batched_roughness_continuity_markov:failed', ...
              '%d test(s) failed.', nFail);
    end
end
