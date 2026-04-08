%% demo_virtualPitches.m
%  Computes and plots virtual pitch (fundamental) salience profiles
%  for a set of example chords.
%
%  Each subplot shows the normalized cross-correlation between the
%  chord's composite spectrum and a harmonic template, plotted against
%  pitch. Peaks indicate strong virtual pitches — candidate
%  fundamentals that are well-supported by the chord's spectral
%  content. Peaks below the lowest chord tone (marked by dashed
%  vertical lines) are the subharmonic virtual pitches that are most
%  characteristic of the chord's identity.
%
%  The six example chords include three 12-TET triads and their
%  just-intonation counterparts, illustrating how mistuning broadens
%  and reduces virtual pitch peaks.
%
%  Uses: virtualPitches, convertPitch
%  (from the Music Perception Toolbox).

%% === User-adjustable parameters ===

% Smoothing width in cents (9-15 are typical; 12 is a good default)
sigma = 12;

% Spectral parameters for the harmonic template
spec = {'harmonic', 36, 'powerlaw', 1};

% Spectral parameters for the chord tones. Set to {} to treat input
% pitches as raw spectral peaks (e.g., from audioPeaks). To model each
% chord tone as a complex tone with the same spectrum as the template:
chordSpec = {'harmonic', 36, 'powerlaw', 1};

% Grid resolution in cents (finer = more accurate but longer output)
resolution = 1;

% Chords as MIDI pitches. Each row: {[pitches], 'label'}.
chordData = {
    [57, 64, 73],             'Open A major'
    [60, 67, 75],             'Open C minor'
    [61, 64, 69]              'A major first inversion'
    [63, 67, 72],             'C minor first inversion'
    [64, 69, 73]              'A major second inversion'
    [67, 72, 75],             'C minor second inversion'
    [57, 64.02, 72.8631],     'Open just A major'
    [60, 67.02, 75.1564],     'Open just C minor'
    [60, 63, 66],             'Close C dim.'
    [60, 61, 62],             'Cluster (C-Db-D)'
    [60, 63.1564, 65.8251]    '5:6:7 dim (just)'
    [57, 61, 65]              'A augmented'
};

% Display range in MIDI note numbers ([] for automatic)
displayRange = [24, 84];

%% === Compute virtual pitch profiles ===

nChords = size(chordData, 1);

% Store results for plotting and summary
results = struct('midi', {}, 'label', {}, 'vp_p', {}, 'vp_w', {});

for i = 1:nChords
    midiPitches = sort(chordData{i, 1});
    label       = chordData{i, 2};

    % Convert MIDI to absolute cents
    p = convertPitch(midiPitches(:), 'midi', 'cents');

    % Compute virtual pitch salience profile
    [vp_p, vp_w] = virtualPitches(p, [], sigma, ...
        'spectrum', spec, ...
        'chordSpectrum', chordSpec, ...
        'resolution', resolution);

    results(i).midi  = midiPitches;
    results(i).label = label;
    results(i).vp_p  = vp_p;
    results(i).vp_w  = vp_w;
end

%% === Plot ===

nCols = 2;
nRows = ceil(nChords / nCols);

figure('Name', 'Virtual pitch profiles', ...
    'Position', [50, 50, 1000, 200 * nRows + 80]);

for i = 1:nChords
    midiPitches = results(i).midi;
    vp_midi     = convertPitch(results(i).vp_p, 'cents', 'midi');
    vp_w        = results(i).vp_w;

    subplot(nRows, nCols, i);
    plot(vp_midi, vp_w, 'LineWidth', 0.8, 'Color', [0.1 0.3 0.7]);

    % Display range
    if isempty(displayRange)
        xlim([midiPitches(1) - 24, midiPitches(end) + 12]);
    else
        xlim(displayRange);
    end
    ylim([0, 1.05 * max(vp_w)]);

    % Mark chord tones with dashed vertical lines
    hold on;
    for j = 1:numel(midiPitches)
        xline(midiPitches(j), '--', 'Color', [0.7 0.2 0.2], ...
            'Alpha', 0.4, 'LineWidth', 0.6);
    end
    hold off;

    xlabel('Pitch (MIDI note number)');
    ylabel('Salience');
    title(sprintf('%s  [%s]', results(i).label, ...
        strjoin(arrayfun(@(x) num2str(x, '%.4g'), midiPitches, ...
        'UniformOutput', false), ', ')));
    xticks(0:12:128);
    grid on;
    set(gca, 'GridAlpha', 0.15);
end

sgtitle(sprintf('Virtual pitch profiles  (\\sigma = %d,  %s)', ...
    sigma, ...
    strjoin(cellfun(@num2str, spec, 'UniformOutput', false), ', ')), ...
    'FontWeight', 'bold');

%% === Print summary ===
% Report the strongest virtual pitch for each chord.

fprintf('\n--- Strongest virtual pitch per chord ---\n');
fprintf('%-25s  %10s  %10s  %10s\n', ...
    'Chord', 'VP (cents)', 'VP (MIDI)', 'Salience');
fprintf('%s\n', repmat('-', 1, 60));

for i = 1:nChords
    [maxW, maxIdx] = max(results(i).vp_w);
    bestCents = results(i).vp_p(maxIdx);
    bestMidi  = convertPitch(bestCents, 'cents', 'midi');

    fprintf('%-25s  %10.1f  %10.2f  %10.3f\n', ...
        results(i).label, bestCents, bestMidi, maxW);
end
fprintf('\n');
