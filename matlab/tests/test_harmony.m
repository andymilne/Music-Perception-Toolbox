%% test_harmony.m — tensorHarmonicity, templateHarmonicity, virtualPitches, spectralEntropy
%
%  Tests for tensorHarmonicity, templateHarmonicity, virtualPitches, spectralEntropy.
%
%  Standalone-runnable; appends to `results` when called from
%  test_mpt.m.

if ~exist('results', 'var')
    results = {};
    standalone = true;
    addpath(fileparts(mfilename('fullpath')));
    clear cleanupDefaults
    cleanupDefaults = mptTestIsolateDefaults(); %#ok<NASGU>
else
    standalone = false;
end



r = roughness(440, 1);
results{end+1,1} = 'roughness: unison = 0';
results{end,2}   = abs(r) < 1e-10;

r = roughness([300, 330], [1, 1]);
results{end+1,1} = 'roughness: positive for nearby freqs';
results{end,2}   = r > 0;

spec = {'harmonic', 24, 'powerlaw', 1};
H_ji = spectralEntropy([0, 386.31, 701.96], [], 12, 'spectrum', spec, 'verbose', false);
H_edo = spectralEntropy([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy: JI < EDO';
results{end,2}   = H_ji < H_edo;

% --- spectralEntropy 2-D batched dispatch (Bundle 2, v2.1+) ---
P_se = [0, 386.31, 701.96; 0, 400, 700; 0, 100, 200];
H_se_batch = spectralEntropy(P_se, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: returns column vector of length nRows';
results{end,2}   = isequal(size(H_se_batch), [3, 1]);

% Per-row matches scalar
H_se_row1 = spectralEntropy(P_se(1, :), [], 12, 'spectrum', spec, 'verbose', false);
H_se_row2 = spectralEntropy(P_se(2, :), [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(H_se_batch(1) - H_se_row1) < 1e-12 ...
                && abs(H_se_batch(2) - H_se_row2) < 1e-12;

% Dedup: two rows that are transpositions of each other give identical entropy
P_se_dup = [0, 400, 700; 100, 500, 800];   % row 2 = row 1 + 100
H_se_dup = spectralEntropy(P_se_dup, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: transposition dedup';
results{end,2}   = abs(H_se_dup(1) - H_se_dup(2)) < 1e-12;

% NaN-padded variable cardinality
P_se_pad = [0, 400, 700, NaN; 0, 1200, NaN, NaN; 0, 300, 600, 900];
H_se_pad = spectralEntropy(P_se_pad, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: NaN-padded cardinality OK';
results{end,2}   = all(~isnan(H_se_pad)) && isequal(size(H_se_pad), [3, 1]);

% All-NaN row returns NaN
P_se_allnan = [0, 400, 700; NaN, NaN, NaN];
H_se_allnan = spectralEntropy(P_se_allnan, [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'spectralEntropy batched: all-NaN row returns NaN';
results{end,2}   = ~isnan(H_se_allnan(1)) && isnan(H_se_allnan(2));

% Verbose printing tests
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for typical fast inputs is silent. The print path itself is exercised
% via the batched-mode tests below and via direct estimateCompTime tests.
outSEScalar = evalc('spectralEntropy([0, 400, 700], [], 12, ''spectrum'', spec, ''verbose'', true);');
results{end+1,1} = 'spectralEntropy scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outSEScalar));

outSEScalarSilent = evalc('spectralEntropy([0, 400, 700], [], 12, ''spectrum'', spec, ''verbose'', false);');
results{end+1,1} = 'spectralEntropy scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outSEScalarSilent));

outSEBatch = evalc('spectralEntropy(P_se, [], 12, ''spectrum'', spec, ''verbose'', true);');
results{end+1,1} = 'spectralEntropy batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outSEBatch));

% --- entropyExpTens batched verbose (Bundle 2) ---
P_ee = [0, 100, 200, 300; 0, 200, 400, 600; 0, 100, 200, 300];
outEEBatch = evalc(['entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ' ...
    '''xMin'', 0, ''xMax'', 600, ''verbose'', true);']);
results{end+1,1} = 'entropyExpTens batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outEEBatch));

outEEBatchSilent = evalc(['entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ' ...
    '''xMin'', 0, ''xMax'', 600, ''verbose'', false);']);
results{end+1,1} = 'entropyExpTens batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outEEBatchSilent));

% Numerical results unchanged by verbose flag
H_ee_v = entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ...
    'xMin', 0, 'xMax', 600, 'verbose', true);
H_ee_q = entropyExpTens(P_ee, [], 12, 1, false, false, 1200, ...
    'xMin', 0, 'xMax', 600, 'verbose', false);
results{end+1,1} = 'entropyExpTens batched: verbose flag does not affect outputs';
results{end,2}   = isequaln(H_ee_v, H_ee_q);

[hMax, hEnt] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hMax in (0,1]';
results{end,2}   = hMax > 0 && hMax <= 1;
results{end+1,1} = 'templateHarmonicity: hEntropy in (0,1]';
results{end,2}   = hEnt > 0 && hEnt <= 1;

% -- templateHarmonicity: hEntropy lower for octave than for cluster --
[~, hEnt_oct] = templateHarmonicity([0, 1200], [], 12, 'verbose', false);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hEntropy octave < cluster';
results{end,2}   = hEnt_oct < hEnt_clu;

% -- templateHarmonicity: hEntropy lower for major triad than for cluster
% (3-note vs 3-note, controlling for cardinality) --
[~, hEnt_maj] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
[~, hEnt_clu] = templateHarmonicity([0, 100, 200], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: hEntropy major triad < cluster';
results{end,2}   = hEnt_maj < hEnt_clu;

spec = {'harmonic', 12, 'powerlaw', 1};
h_uni = tensorHarmonicity([0, 0], [], 12, 'spectrum', spec, 'verbose', false);
h_tri = tensorHarmonicity([0, 600], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: unison > tritone';
results{end,2}   = h_uni > h_tri;

% -- tensorHarmonicity: ordered ranking
% octave (2:1) > perfect 5th (3:2) > major triad (4:5:6) > minor triad --
h_oct  = tensorHarmonicity([0, 1200],     [], 12, 'spectrum', spec, 'verbose', false);
h_p5   = tensorHarmonicity([0, 700],      [], 12, 'spectrum', spec, 'verbose', false);
h_maj  = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
h_min  = tensorHarmonicity([0, 300, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: octave > P5 > major > minor';
results{end,2}   = (h_oct > h_p5) && (h_p5 > h_maj) && (h_maj > h_min);

[vp_p, vp_w] = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches: non-empty';
results{end,2}   = numel(vp_p) > 0;
results{end+1,1} = 'virtualPitches: lengths match';
results{end,2}   = numel(vp_p) == numel(vp_w);

% -- virtualPitches: peak of a single pitch sits at the pitch itself --
[vp_p1, vp_w1] = virtualPitches(400, [], 12, 'verbose', false);
[~, i_max1] = max(vp_w1);
results{end+1,1} = 'virtualPitches: single pitch peak at the pitch';
results{end,2}   = abs(vp_p1(i_max1) - 400) < 5;

% -- virtualPitches: peak of an octave dyad at the lower note
% (partials 1 and 2 of a template at 0 align with both chord notes) --
[vp_p2, vp_w2] = virtualPitches([0, 1200], [], 12, 'verbose', false);
[~, i_max2] = max(vp_w2);
results{end+1,1} = 'virtualPitches: octave peak at lower note';
results{end,2}   = abs(vp_p2(i_max2)) < 5;

% --- v2.1 unified dispatch: harmony wrappers batched mode ---

% tensorHarmonicity: 2-D matrix dispatch returns column vector
P_h = [0, 1200, 0, 0;     % unison-with-octave (4-pitch); cardinality 4
       0, 700, 0, 0;       % open-fifth-doubled (4-pitch); cardinality 4
       0, 400, 700, 0];    % major triad (3-pitch with NaN pad — wait, 0 is a pitch)
% The above is ambiguous because 0 is a valid pitch. Use NaN padding instead.
P_th = [0, 1200,  NaN;
        0, 400,   700;
        0, 300,   700];
spec_th = {'harmonic', 12, 'powerlaw', 1};
h_th = tensorHarmonicity(P_th, [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: returns column vector of correct length';
results{end,2}   = isnumeric(h_th) && isequal(size(h_th), [3, 1]);

% Matches scalar dispatch row-by-row (NaN dropped per row)
h_oct  = tensorHarmonicity([0, 1200],     [], 12, 'spectrum', spec_th, 'verbose', false);
h_maj3 = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
h_min3 = tensorHarmonicity([0, 300, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(h_th(1) - h_oct)  < 1e-12 ...
                   && abs(h_th(2) - h_maj3) < 1e-12 ...
                   && abs(h_th(3) - h_min3) < 1e-12;

% Major > minor preserved across batched rows (existing scalar property)
results{end+1,1} = 'tensorHarmonicity batched: major > minor (octave > both)';
results{end,2}   = h_th(1) > h_th(2) && h_th(2) > h_th(3);

% Row with fewer than 2 valid pitches returns NaN
P_th_short = [0, 400, 700;  NaN, NaN, NaN; 0, NaN, NaN];   % row 2 empty, row 3 has 1
h_th_short = tensorHarmonicity(P_th_short, [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: insufficient pitches return NaN';
results{end,2}   = ~isnan(h_th_short(1)) && isnan(h_th_short(2)) && isnan(h_th_short(3));

% --- tensorHarmonicity verbose / estimateCompTime integration (v2.1.1+) ---
% Scalar verbose=true forwards to buildExpTens, which prints its own estimate.
outScalarVerb = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', true);']);
results{end+1,1} = 'tensorHarmonicity scalar: verbose=true prints something';
results{end,2}   = ~isempty(strtrim(outScalarVerb));

outScalarSilent = evalc(['tensorHarmonicity([0, 400, 700], [], 12, ' ...
    '''spectrum'', spec, ''verbose'', false);']);
results{end+1,1} = 'tensorHarmonicity scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outScalarSilent));

outBatchVerb = evalc(['tensorHarmonicity(P_th, [], 12, ''spectrum'', spec_th, ' ...
    '''verbose'', true);']);
results{end+1,1} = 'tensorHarmonicity batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outBatchVerb));

outBatchSilent = evalc(['tensorHarmonicity(P_th, [], 12, ''spectrum'', spec_th, ' ...
    '''verbose'', false);']);
results{end+1,1} = 'tensorHarmonicity batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outBatchSilent));

[hA] = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', true);
[hB] = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity: verbose flag does not affect outputs';
results{end,2}   = (hA == hB);

% templateHarmonicity batched
[hMax_b, hEnt_b] = templateHarmonicity(P_th, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: hMax shape';
results{end,2}   = isequal(size(hMax_b), [3, 1]);
results{end+1,1} = 'templateHarmonicity batched: hEntropy shape';
results{end,2}   = isequal(size(hEnt_b), [3, 1]);

% Matches scalar row-by-row
[hMax_oct,  hEnt_oct]  = templateHarmonicity([0, 1200],     [], 12, 'verbose', false);
[hMax_maj3, hEnt_maj3] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: matches scalar dispatch row-by-row';
results{end,2}   = abs(hMax_b(1) - hMax_oct)  < 1e-12 ...
                   && abs(hEnt_b(1) - hEnt_oct)  < 1e-12 ...
                   && abs(hMax_b(2) - hMax_maj3) < 1e-12 ...
                   && abs(hEnt_b(2) - hEnt_maj3) < 1e-12;

% --- templateHarmonicity batched: chord-side dedup (v2.2+) ---
% Two rows that are transpositions of each other must produce
% identical hMax and hEntropy (template-harmonicity transposes
% internally so any shift cancels). Same for permutations.
P_th_trans = [0, 400, 700; 100, 500, 800; 1200, 1600, 1900];
[hMaxT_th, hEntT_th] = templateHarmonicity(P_th_trans, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: transposition dedup (hMax)';
results{end,2}   = abs(hMaxT_th(1) - hMaxT_th(2)) < 1e-12 ...
                && abs(hMaxT_th(1) - hMaxT_th(3)) < 1e-12;
results{end+1,1} = 'templateHarmonicity batched: transposition dedup (hEntropy)';
results{end,2}   = abs(hEntT_th(1) - hEntT_th(2)) < 1e-12 ...
                && abs(hEntT_th(1) - hEntT_th(3)) < 1e-12;

P_th_perm = [0, 400, 700; 700, 0, 400; 400, 700, 0];
[hMaxP_th, hEntP_th] = templateHarmonicity(P_th_perm, [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity batched: permutation dedup (hMax)';
results{end,2}   = abs(hMaxP_th(1) - hMaxP_th(2)) < 1e-12 ...
                && abs(hMaxP_th(1) - hMaxP_th(3)) < 1e-12;
results{end+1,1} = 'templateHarmonicity batched: permutation dedup (hEntropy)';
results{end,2}   = abs(hEntP_th(1) - hEntP_th(2)) < 1e-12 ...
                && abs(hEntP_th(1) - hEntP_th(3)) < 1e-12;
clear P_th_trans P_th_perm hMaxT_th hEntT_th hMaxP_th hEntP_th

% --- templateHarmonicity verbose / estimateCompTime integration (v2.1.1+) ---
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for fast inputs is silent.

% Scalar verbose=true is silent for typical fast inputs (sub-half-sec)
outScalarVerb = evalc('templateHarmonicity([0, 400, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outScalarVerb));

% Scalar verbose=false suppresses output entirely
outScalarSilent = evalc('templateHarmonicity([0, 400, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'templateHarmonicity scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outScalarSilent));

% Default verbose is true but fast scalar still silent under threshold
outDefault = evalc('templateHarmonicity([0, 400, 700], [], 12);');
results{end+1,1} = 'templateHarmonicity scalar: default verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outDefault));

% Batched verbose=true silent for fast call (new contract: 10s threshold)
outBatchVerb = evalc('templateHarmonicity([0, 400, 700; 0, 300, 700; 0, 300, 600], [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outBatchVerb));

% Batched verbose=false silent
outBatchSilent = evalc('templateHarmonicity([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'templateHarmonicity batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outBatchSilent));

% Numerical results unaffected by verbose flag
[hMaxA, hEntA] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', true);
[hMaxB, hEntB] = templateHarmonicity([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'templateHarmonicity: verbose flag does not affect outputs';
results{end,2}   = (hMaxA == hMaxB) && (hEntA == hEntB);

% All-NaN batched input does not crash and gives NaN output
P_allnan = NaN(3, 3);
outAllNaN = evalc('[hMax_n, hEnt_n] = templateHarmonicity(P_allnan, [], 12, ''verbose'', true);');
results{end+1,1} = 'templateHarmonicity batched: all-NaN rows yield NaN, no crash';
% Need to capture outputs — re-run without evalc to get them
[hMax_n, hEnt_n] = templateHarmonicity(P_allnan, [], 12, 'verbose', false);
results{end,2}   = isequal(size(hMax_n), [3, 1]) && all(isnan(hMax_n)) ...
                && isequal(size(hEnt_n), [3, 1]) && all(isnan(hEnt_n));

% virtualPitches batched: cell-of-arrays output
[vp_pcell, vp_wcell] = virtualPitches(P_th, [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches batched: vp_p is 1-by-nRows cell';
results{end,2}   = iscell(vp_pcell) && isequal(size(vp_pcell), [1, 3]);
results{end+1,1} = 'virtualPitches batched: vp_w is 1-by-nRows cell';
results{end,2}   = iscell(vp_wcell) && isequal(size(vp_wcell), [1, 3]);

% Each cell entry matches scalar dispatch
[vp_p_oct,  vp_w_oct]  = virtualPitches([0, 1200],     [], 12, 'verbose', false);
[vp_p_maj,  vp_w_maj]  = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches batched: cell entries match scalar dispatch';
results{end,2}   = numel(vp_pcell{1}) == numel(vp_p_oct) ...
                   && max(abs(vp_pcell{1} - vp_p_oct)) < 1e-12 ...
                   && numel(vp_pcell{2}) == numel(vp_p_maj) ...
                   && max(abs(vp_pcell{2} - vp_p_maj)) < 1e-12;

% NaN-padded rows produce same outputs as unpadded
P_clean = [0, 1200; 0, 400; 0, 300];   % no NaN padding (cardinality 2)
P_padded = [0, 1200, NaN; 0, 400, NaN; 0, 300, NaN];
spec_simple = {'harmonic', 12, 'powerlaw', 1};
h_clean  = tensorHarmonicity(P_clean,  [], 12, 'spectrum', spec_simple, 'verbose', false);
h_padded = tensorHarmonicity(P_padded, [], 12, 'spectrum', spec_simple, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity batched: NaN-padded rows match unpadded';
results{end,2}   = max(abs(h_clean - h_padded)) < 1e-12;

% Scalar (vector) input still dispatches to scalar path
h_scalar_check = tensorHarmonicity([0, 400, 700], [], 12, 'spectrum', spec_th, 'verbose', false);
results{end+1,1} = 'tensorHarmonicity scalar: row vector still works (backward compat)';
results{end,2}   = isscalar(h_scalar_check) && abs(h_scalar_check - h_maj3) < 1e-14;

% --- virtualPitches verbose / estimateCompTime integration (v2.1.1+) ---
% Note (commit 14+): estimateCompTime default minPrintSec is 10,
% so verbose=true
% for fast inputs is silent.
outVPScalarVerb = evalc('virtualPitches([0, 400, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'virtualPitches scalar: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPScalarVerb));

outVPScalarSilent = evalc('virtualPitches([0, 400, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'virtualPitches scalar: verbose=false silent';
results{end,2}   = isempty(strtrim(outVPScalarSilent));

outVPDefault = evalc('virtualPitches([0, 400, 700], [], 12);');
results{end+1,1} = 'virtualPitches scalar: default verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPDefault));

outVPBatchVerb = evalc('virtualPitches([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', true);');
results{end+1,1} = 'virtualPitches batched: verbose=true silent for fast call';
results{end,2}   = isempty(strtrim(outVPBatchVerb));

outVPBatchSilent = evalc('virtualPitches([0, 400, 700; 0, 300, 700], [], 12, ''verbose'', false);');
results{end+1,1} = 'virtualPitches batched: verbose=false silent';
results{end,2}   = isempty(strtrim(outVPBatchSilent));

[vp_pA, vp_wA] = virtualPitches([0, 400, 700], [], 12, 'verbose', true);
[vp_pB, vp_wB] = virtualPitches([0, 400, 700], [], 12, 'verbose', false);
results{end+1,1} = 'virtualPitches: verbose flag does not affect outputs';
results{end,2}   = isequal(vp_pA, vp_pB) && isequal(vp_wA, vp_wB);


%% ---- Standalone summary ----

if standalone
    nPass = sum([results{:, 2}]);
    nFail = size(results, 1) - nPass;
    for i = 1:size(results, 1)
        if results{i, 2}
            fprintf('  PASS  %s\n', results{i, 1});
        else
            fprintf('  FAIL  %s\n', results{i, 1});
        end
    end
    fprintf('\n=== test_harmony: %d passed, %d failed (of %d) ===\n\n', ...
        nPass, nFail, nPass + nFail);
    clear cleanupDefaults
    if nFail > 0
        error('test_harmony:failed', '%d test(s) failed.', nFail);
    end
end
