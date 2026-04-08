%% demo_audioAnalysis.m
%  Extract spectral peaks from audio files and compute perceptual features.
%
%  Demonstrates the workflow for analysing real audio: peak extraction
%  via audioPeaks, then spectral similarity (SPCS), harmonicity,
%  spectral entropy, roughness, and virtual pitches — all without
%  spectral enrichment, since the extracted peaks already represent
%  the full spectrum.
%
%  Two peak-extraction passes are shown:
%    1. No smoothing — appropriate for steady-state sounds (piano, oboe)
%       but produces many spurious peaks for sounds with vibrato or
%       frequency jitter (violin, complex music).
%    2. With Gaussian smoothing (sigmaPeaks cents) — collapses
%       vibrato-spread energy into single peaks, giving a cleaner
%       representation for all sources.
%
%  Perceptual features are computed from the smoothed peaks.
%
%  The audio files are in the audio/ subfolder.
%
%  Uses: audioPeaks, convertPitch, spectralEntropy, templateHarmonicity,
%        roughness, virtualPitches, cosSimExpTens
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

sigma      = 12;    % Gaussian smoothing width for perceptual measures
sigmaPeaks = 12;    % Gaussian smoothing width for peak extraction

% Audio files to analyse (relative to the toolbox root)
audioFiles = {
    '../audio/piano_C4.wav',                'Piano C4'
    '../audio/violin_A4.wav',               'Violin A4'
    '../audio/oboe_A4.wav',                 'Oboe A4'
    '../audio/Piano_Emin.wav',              'Piano E minor'
    '../audio/Piano_G7_3rd_inversion.wav',  'Piano G7 (3rd inv.)'
    '../audio/Piano_Cmin_open.wav',         'Piano C minor (open)'
    '../audio/music_sample.wav',            'Music sample'
};

nFiles = size(audioFiles, 1);

%% === Pass 1: Extract peaks without smoothing ===

fprintf('=== Pass 1: Peak extraction (no smoothing) ===\n\n');

rawPeakCounts = zeros(nFiles, 1);
validFiles    = false(nFiles, 1);

for i = 1:nFiles
    filepath = audioFiles{i, 1};
    label    = audioFiles{i, 2};

    if ~isfile(filepath)
        fprintf('  %s: file not found (%s), skipping.\n\n', label, filepath);
        continue;
    end

    [f, ~] = audioPeaks(filepath);
    rawPeakCounts(i) = numel(f);
    validFiles(i)    = true;
    fprintf('\n');
end

%% === Pass 2: Extract peaks with smoothing ===

fprintf('=== Pass 2: Peak extraction (sigma = %d cents) ===\n\n', sigmaPeaks);

results = struct('label', {}, 'f', {}, 'w', {}, 'p', {});

for i = 1:nFiles
    filepath = audioFiles{i, 1};
    label    = audioFiles{i, 2};

    if ~validFiles(i)
        continue;
    end

    [f, w] = audioPeaks(filepath, 'sigma', sigmaPeaks);
    p = convertPitch(f, 'hz', 'cents');

    idx = numel(results) + 1;
    results(idx).label = label;
    results(idx).f = f;
    results(idx).w = w;
    results(idx).p = p;
    results(idx).rawPeakCount = rawPeakCounts(i);

    fprintf('\n');
end

nResults = numel(results);

if nResults == 0
    fprintf('No audio files found. Check the audio/ folder.\n');
    return;
end

%% === Peak count comparison ===

fprintf('=== Peak count comparison ===\n\n');
fprintf('%-25s  %10s  %10s\n', 'Sound', 'Unsmoothed', 'Smoothed');
fprintf('%s\n', repmat('-', 1, 50));

for i = 1:nResults
    fprintf('%-25s  %10d  %10d\n', ...
        results(i).label, results(i).rawPeakCount, numel(results(i).f));
end

%% === Single-file features (smoothed peaks, no spectral enrichment) ===

fprintf('\n=== Single-file features (smoothed peaks, sigma = %d) ===\n\n', sigma);
fprintf('%-25s  %5s  %8s  %8s  %8s  %8s\n', ...
    'Sound', 'Peaks', 'specEnt', 'hMax', 'hEnt', 'Rough');
fprintf('%s\n', repmat('-', 1, 72));

for i = 1:nResults
    f = results(i).f;
    w = results(i).w;
    p = results(i).p;

    H = spectralEntropy(p, w, sigma);
    [hMax, hEnt] = templateHarmonicity(p, w, sigma);
    r = roughness(f, w);

    results(i).specEnt = H;
    results(i).hMax    = hMax;
    results(i).hEnt    = hEnt;
    results(i).rough   = r;

    fprintf('%-25s  %5d  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        results(i).label, numel(f), H, hMax, hEnt, r);
end

%% === Pairwise spectral pitch class similarity ===

fprintf('\n=== Pairwise SPCS (smoothed peaks, sigma = %d) ===\n\n', sigma);

% Header
fprintf('%25s  ', '');
for j = 1:nResults
    lbl = results(j).label;
    if numel(lbl) > 12, lbl = lbl(1:12); end
    fprintf('%-12s', lbl);
end
fprintf('\n');

for i = 1:nResults
    fprintf('%-25s  ', results(i).label);
    for j = 1:nResults
        if j < i
            fprintf('%12s', '');
        elseif j == i
            fprintf('%-12s', '1.000');
        else
            s = cosSimExpTens(results(i).p, results(i).w, ...
                              results(j).p, results(j).w, ...
                              sigma, 1, false, true, 1200, ...
                              'verbose', false);
            fprintf('%-12.3f', s);
        end
    end
    fprintf('\n');
end

%% === Virtual pitches ===

fprintf('\n=== Strongest virtual pitch per sound (smoothed peaks) ===\n\n');
fprintf('%-25s  %10s  %10s  %10s\n', ...
    'Sound', 'VP (cents)', 'VP (MIDI)', 'Salience');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:nResults
    [vp_p, vp_w] = virtualPitches(results(i).p, results(i).w, sigma);
    [maxW, maxIdx] = max(vp_w);
    bestCents = vp_p(maxIdx);
    bestMidi  = convertPitch(bestCents, 'cents', 'midi');

    fprintf('%-25s  %10.1f  %10.2f  %10.3f\n', ...
        results(i).label, bestCents, bestMidi, maxW);
end

fprintf('\nDone.\n');
