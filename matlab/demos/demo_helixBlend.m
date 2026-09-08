%% demo_helixBlend.m
%  Helix blend: routing pitch through two groups of a MAET.
%
%  Demonstrates a multi-attribute expectation tensor pattern in which
%  the same pitch values are routed simultaneously through a periodic
%  pitch-class group and a linear pitch-height group. Sweeping the
%  pitch-height-group sigma while holding the pitch-class-group sigma fixed
%  morphs the similarity profile of a motif against a longer stream
%  from
%
%    * "matches every octave-displaced recurrence equally"  (large sigma_ph)
%    * through graded octave tolerance                      (medium)
%    * to "matches only the same-height recurrence"         (small sigma_ph).
%
%  Equivalence with Shepard's model. The factored Gaussian
%
%      exp(- d_pc(p1,p2)^2 / (2 sigma_pc^2))
%    * exp(-  (p1 - p2)^2  / (2 sigma_ph^2))
%
%  is equivalent to a Gaussian kernel of width sigma = sigma_pc on the
%  pitch-class-cum-height cylinder with stretch
%  h = sigma_pc/sigma_ph. Shepard's helix itself has no built-in
%  smoothing; this MAET pattern adds it, parametrised naturally in two
%  pitch-domain sigma values.
%
%  Two technical points. First, the sigmas are density widths (MPT's
%  convention throughout the toolbox); the pairwise inner-product kernel
%  between two smeared events has effective standard deviation
%  sqrt(2)*sigma. Second, the MAET pitch-class group uses shortest-arc
%  distance, so the geometry is the pc cylinder rather than the literal
%  3-D Shepard helix (a Euclidean embedding that uses chord distance).
%  At sigma_pc values typical of tonal perception the two are
%  indistinguishable.
%
%  Two parts:
%
%    Part 1. Synthetic. A three-note C-major motif stated at four
%            heights, with non-pitch-class-overlapping filler between
%            instances.
%
%    Part 2. Fugal texture in C minor (BWV 847-inspired, stylised; not
%            transcribed from the score). A six-note subject stated in
%            bass, alto, and soprano, with short counter-material
%            between entries.
%
%  Each part produces three stacked panels:
%    (a) the event stream,
%    (b) a similarity heatmap over (time offset, sigma_ph),
%    (c) three overlaid profile curves at representative sigma_ph
%        values.
%
%  Uses: windowedSimilarity (event weighting), transformAttributes.

clear; clc; close all;

%% === User parameters ===

% -- Common --
SIG_PC           = 30;                               % pc group sigma (cents)
SIG_PH_SWEEP    = logspace(log10(100), log10(8000), 25);
SIG_PH_PROFILES = [200, 600, 3000];                 % three overlaid profiles
WIN_MIX          = 0.5;                              % rectangular x Gaussian

% -- Part 1 (synthetic) --
SIG_TIME_1       = 0.10;                             % sec
WIN_SIZE_TIME_1  = 6.0;                              % window sd in units of sigma_time
OFFSETS_1        = -0.5 : 0.02 : 12.5;

% -- Part 2 (fugal texture) --
SIG_TIME_2       = 0.08;
WIN_SIZE_TIME_2  = 12.0;
OFFSETS_2        = -0.5 : 0.02 : 8.5;

%% === Part 1: synthetic motif at four heights ===

fprintf('Part 1: synthetic motif at four heights.\n');
[ctx1_midi, ctx1_t, q1_midi, q1_t, motif_idx1, motif_cent1] = buildPart1Stream();
ctx1_cents = transformAttributes(ctx1_midi, [], {'midi', 'cents'});
q1_cents   = transformAttributes(q1_midi,   [], {'midi', 'cents'});

heat1 = sweepProfiles(q1_cents, q1_t, ctx1_cents, ctx1_t, ...
    SIG_PC, SIG_PH_SWEEP, SIG_TIME_1, WIN_SIZE_TIME_1, WIN_MIX, ...
    OFFSETS_1, true);
prof1 = sweepProfiles(q1_cents, q1_t, ctx1_cents, ctx1_t, ...
    SIG_PC, SIG_PH_PROFILES, SIG_TIME_1, WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1);
peak1 = motif_cent1 - mean(q1_t);
reportPeaks(prof1, OFFSETS_1, SIG_PH_PROFILES, peak1);

figure('Name', 'Helix blend: synthetic', 'Position', [80, 80, 980, 800]);
plotPart(gcf, 'Helix blend (synthetic): C-E-G at four heights', ...
    ctx1_midi, ctx1_t, motif_idx1, peak1, mean(q1_t), ...
    heat1, prof1, OFFSETS_1, SIG_PH_SWEEP, SIG_PH_PROFILES, ...
    'motif events (C-E-G)', 'filler events');

%% === Part 2: fugal texture (BWV 847-inspired, stylised) ===

fprintf('Part 2: fugal texture (BWV 847-inspired, stylised).\n');
[ctx2_midi, ctx2_t, q2_midi, q2_t, subj_idx2, subj_cent2] = buildPart2Stream();
ctx2_cents = transformAttributes(ctx2_midi, [], {'midi', 'cents'});
q2_cents   = transformAttributes(q2_midi,   [], {'midi', 'cents'});

heat2 = sweepProfiles(q2_cents, q2_t, ctx2_cents, ctx2_t, ...
    SIG_PC, SIG_PH_SWEEP, SIG_TIME_2, WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2);
prof2 = sweepProfiles(q2_cents, q2_t, ctx2_cents, ctx2_t, ...
    SIG_PC, SIG_PH_PROFILES, SIG_TIME_2, WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2);
peak2 = subj_cent2 - mean(q2_t);
reportPeaks(prof2, OFFSETS_2, SIG_PH_PROFILES, peak2);

figure('Name', 'Helix blend: fugal texture', 'Position', [120, 120, 980, 800]);
plotPart(gcf, ['Helix blend (BWV 847-inspired, stylised): ' ...
               'subject in bass, alto, soprano'], ...
    ctx2_midi, ctx2_t, subj_idx2, peak2, mean(q2_t), ...
    heat2, prof2, OFFSETS_2, SIG_PH_SWEEP, SIG_PH_PROFILES, ...
    'subject events', 'counter-material');

% =====================================================================
%  Local functions
% =====================================================================

function reportPeaks(prof, offsets, sigma_phs, true_peaks)
%REPORTPEAKS  Print, per profile sigma_ph, the similarity at each true
%   statement offset: as sigma_ph widens, octave-displaced statements
%   rise from near zero towards the same-height value of 1.
    for i = 1:numel(sigma_phs)
        vals = zeros(1, numel(true_peaks));
        for k = 1:numel(true_peaks)
            [~, j] = min(abs(offsets - true_peaks(k)));
            vals(k) = prof(i, j);
        end
        fprintf('  sigma_ph = %6.0f cents: similarity at the statements = %s\n', ...
            sigma_phs(i), strjoin(arrayfun(@(v) sprintf('%.4f', v), vals, 'UniformOutput', false), ', '));
    end
end

function pm = helixPreMaet(pitch_cents, time_sec, sigma_pc, sigma_time)
%HELIXPREMAET  The same pitch values routed through two attributes, plus time.
%
%   Attributes: (pitch, pitch, time), read as (pc, ph, time): the first
%   pitch copy is periodic at 1200 cents, the second and the time axis
%   are linear. All r = 1.
%
%   The pre-MAET carries its own geometry, so nothing has to be threaded
%   alongside it. The pitch-height width is left at NaN -- NA, the value
%   the sweep supplies -- since it is the one parameter that varies and
%   no baseline for it would be honest.
    p = pitch_cents(:).';
    t = time_sec(:).';
    n = numel(p);
    specs = {struct('name', 'pitch class',  'r', 1, 'rel', false, ...
                    'sym', true, 'sigma', sigma_pc,   'isPer', true, ...
                    'period', 1200), ...
             struct('name', 'pitch height', 'r', 1, 'rel', false, ...
                    'sym', true, 'sigma', NaN,        'isPer', false, ...
                    'period', 0), ...
             struct('name', 'time',         'r', 1, 'rel', false, ...
                    'sym', true, 'sigma', sigma_time, 'isPer', false, ...
                    'period', 0)};
    pm = preMaet({p, p, t}, {ones(1, n), ones(1, n), ones(1, n)}, specs);
end

function prof = sweepProfiles(q_cents, q_t, c_cents, c_t, ...
                              sigma_pc, sigma_ph_values, sigma_time, ...
                              win_size_time, win_mix, offsets, showInput)
%SWEEPPROFILES  Return a length(sigma_ph_values) x length(offsets) array
%   of windowed-similarity profiles (a cross-correlation of the query
%   against the time-windowed context).
%
%   Windowing is event weighting: at each sweep position the window,
%   centred on that position along the time axis, multiplies the
%   per-event weights of the context before its density is built, and
%   the query is translated so that its time centroid lands on the same
%   position. The window has standard deviation win_size_time * sigma_time
%   and shape win_mix (0 Gaussian, 1 rectangular).

    % Acquire a top-level dispatch scope for the duration of the
    % per-sigma_ph loop, so the dispatch-announce throttle deduplicates
    % the cosine path's announce across the sweep rather than re-emitting
    % it per iteration. See internal.dispatchScope.
    guard = internal.dispatchScope(); %#ok<NASGU>

    TIME = 3;                                    % the swept (window) axis

    % The window family has fixed variance sd^2 for every shape; the
    % width argument is the rectangle-equivalent full width 2*sqrt(3)*sd.
    sd_time = win_size_time * sigma_time;
    contextWindow = {win_mix, 2 * sqrt(3) * sd_time};

    % Sweep positions are absolute times on the context axis; the plotted
    % offset is the position relative to the query's time centroid.
    centres = offsets(:).' + mean(q_t);

    % Only the pitch-height width varies across the sweep, so the two
    % pre-MAETs are built once and each call names that one parameter.
    % A selective override -- the entries left empty keep what the spec
    % carries -- says exactly that, and the pre-MAETs are unchanged by
    % it. The table below shows sigma = NA on the swept attribute, the
    % value each call supplies.
    pmQ = helixPreMaet(q_cents, q_t, sigma_pc, sigma_time);
    pmC = helixPreMaet(c_cents, c_t, sigma_pc, sigma_time);

    if nargin >= 11 && showInput
        showPreMaet(pmC, 'maxEvents', 4);
        fprintf('\n');
    end

    nS = numel(sigma_ph_values);
    prof = zeros(nS, numel(offsets));
    for i = 1:nS
        prof(i, :) = windowedSimilarity(pmC, pmQ, centres, ...
            'sigma', {[], sigma_ph_values(i), []}, ...
            'windowAttr', TIME, 'dropWindowAttr', false, ...
            'contextWindow', contextWindow, 'locate', 'centroid', ...
            'normalize', 'oneSidedDenom', 'verbose', false);
    end
end

function [ctx_midi, ctx_t, q_midi, q_t, motif_idx, motif_cent] = buildPart1Stream()
%BUILDPART1STREAM  Three-note motif at four heights with non-overlapping filler.
    motif_midi_ref  = [60, 64, 67];        % C4 E4 G4
    filler_midi_ref = [62, 65, 69];        % D4 F4 A4 (disjoint pc's)
    heights_st         = [0, 12, -12, 24];
    dt              = 0.5;

    ctx_midi  = [];
    ctx_t     = [];
    motif_idx = [];
    t = 0;
    for k = 1:numel(heights_st)
        shift_st = heights_st(k);
        for fp = filler_midi_ref
            ctx_midi(end+1) = fp + shift_st; %#ok<AGROW>
            ctx_t(end+1)    = t;             %#ok<AGROW>
            t = t + dt;
        end
        for mp = motif_midi_ref
            motif_idx(end+1) = numel(ctx_midi) + 1; %#ok<AGROW>
            ctx_midi(end+1) = mp + shift_st; %#ok<AGROW>
            ctx_t(end+1)    = t;             %#ok<AGROW>
            t = t + dt;
        end
    end

    q_midi = motif_midi_ref;
    q_t    = (0 : numel(motif_midi_ref) - 1) * dt;

    per_entry = numel(motif_midi_ref);
    motif_cent = zeros(1, numel(heights_st));
    for k = 1:numel(heights_st)
        ii = motif_idx((k-1)*per_entry + 1 : k*per_entry);
        motif_cent(k) = mean(ctx_t(ii));
    end
end

function [ctx_midi, ctx_t, q_midi, q_t, subj_idx, subj_cent] = buildPart2Stream()
%BUILDPART2STREAM  Six-note subject at three heights, short counter-material between.
    subj_ref = [60, 63, 65, 63, 62, 60];       % C Eb F Eb D C
    cnt1     = [57, 55, 53];                   % A3 G3 F3
    cnt2     = [74, 72, 70];                   % D5 C5 Bb4
    dt       = 0.30;
    gap      = 0.30;

    entries_st = [-12, 0, 12];

    ctx_midi  = [];
    ctx_t     = [];
    subj_idx  = [];
    t = 0;
    for k = 1:numel(entries_st)
        shift_st = entries_st(k);
        idx0 = numel(ctx_midi) + 1;
        for mp = subj_ref
            ctx_midi(end+1) = mp + shift_st; %#ok<AGROW>
            ctx_t(end+1)    = t;             %#ok<AGROW>
            t = t + dt;
        end
        subj_idx(end+1 : end + numel(subj_ref)) = idx0 : idx0 + numel(subj_ref) - 1;
        t = t - dt;                                % undo trailing increment
        if k < numel(entries_st)
            t = t + gap;
            if k == 1, cnt = cnt1; else, cnt = cnt2; end
            for mp = cnt
                ctx_midi(end+1) = mp; %#ok<AGROW>
                ctx_t(end+1)    = t;  %#ok<AGROW>
                t = t + dt;
            end
            t = t + gap;
        end
    end

    q_midi = subj_ref;
    q_t    = (0 : numel(subj_ref) - 1) * dt;

    per_entry = numel(subj_ref);
    subj_cent = zeros(1, numel(entries_st));
    for k = 1:numel(entries_st)
        ii = subj_idx((k-1)*per_entry + 1 : k*per_entry);
        subj_cent(k) = mean(ctx_t(ii));
    end
end

function plotPart(fig, suptitle_str, ctx_midi, ctx_t, marker_idx, ...
                  peak_offsets, query_centroid_t, ...
                  heat, prof, offsets, sigma_ph_sweep, sigma_ph_profiles, ...
                  label_marked, label_unmarked)
%PLOTPART  All three panels share the "query offset" x-axis:
%
%      offset = context_time - query_centroid_time.
%
%  Under this convention, each marker (motif/subject) occurrence in
%  panel (a) sits at the same x-coordinate as its corresponding peak
%  in panels (b) and (c).
%
%  The layout uses explicit figure-normalised positions throughout so
%  that the colorbar and legends do not steal width from the main
%  panels (which would misalign their x-axes).
    figure(fig);
    is_marked = false(1, numel(ctx_t));
    is_marked(marker_idx) = true;
    ctx_x = ctx_t - query_centroid_t;

    % -- Figure layout (figure-normalised coordinates) --
    LEFT    = 0.08;
    PLOT_R  = 0.80;                    % right edge of plot columns
    CBAR_L  = PLOT_R + 0.02;           % colourbar left
    CBAR_W  = 0.018;
    LEG_L   = PLOT_R + 0.015;          % legend left (small gap from plot;
                                       %   legends sit on rows that don't
                                       %   have the colourbar)
    LEG_W   = 0.17;

    ROW_A_Y = 0.785;   ROW_A_H = 0.135;
    ROW_B_Y = 0.300;   ROW_B_H = 0.445;
    ROW_C_Y = 0.080;   ROW_C_H = 0.175;

    PLOT_W = PLOT_R - LEFT;

    ax1 = axes(fig, 'Position', [LEFT, ROW_A_Y, PLOT_W, ROW_A_H]);
    ax2 = axes(fig, 'Position', [LEFT, ROW_B_Y, PLOT_W, ROW_B_H]);
    ax3 = axes(fig, 'Position', [LEFT, ROW_C_Y, PLOT_W, ROW_C_H]);

    % (a) Event stream (MIDI y-axis)
    hold(ax1, 'on'); grid(ax1, 'on'); box(ax1, 'on');
    scatter(ax1, ctx_x(~is_marked), ctx_midi(~is_marked), 40, ...
        'MarkerFaceColor', [0.85, 0.85, 0.85], ...
        'MarkerEdgeColor', [0.40, 0.40, 0.40], 'LineWidth', 0.6);
    scatter(ax1, ctx_x(is_marked), ctx_midi(is_marked), 72, ...
        'MarkerFaceColor', [0.72, 0.19, 0.19], ...
        'MarkerEdgeColor', [0.33, 0.13, 0.13], 'LineWidth', 0.6);
    for po = peak_offsets
        xline(ax1, po, ':', 'Color', [0.53, 0.53, 0.53], 'LineWidth', 0.6);
    end
    ylabel(ax1, 'MIDI pitch');
    legend(ax1, {label_unmarked, label_marked}, ...
        'Position', [LEG_L, ROW_A_Y + ROW_A_H - 0.055, LEG_W, 0.045], ...
        'FontSize', 8, 'Box', 'off');
    title(ax1, '(a) Event stream');
    xlim(ax1, [offsets(1), offsets(end)]);

    % (b) Heatmap
    cmax = max(max(heat(:)), 1e-3);
    y_idx = 1:numel(sigma_ph_sweep);
    imagesc(ax2, offsets, y_idx, heat);
    caxis(ax2, [0, cmax]);
    set(ax2, 'YDir', 'normal');
    colormap(ax2, parula);

    log_sw = log(sigma_ph_sweep);
    cand = [100, 200, 500, 1000, 2000, 5000];
    ytick_vals = cand(cand >= min(sigma_ph_sweep) & cand <= max(sigma_ph_sweep));
    ytick_pos  = interp1(log_sw, y_idx, log(ytick_vals));
    set(ax2, 'YTick', ytick_pos, ...
        'YTickLabel', arrayfun(@(v) sprintf('%g', v), ytick_vals, ...
                               'UniformOutput', false));

    hold(ax2, 'on');
    for po = peak_offsets
        xline(ax2, po, '--', 'Color', 'w', 'LineWidth', 0.7);
    end
    for spv = sigma_ph_profiles
        spv_idx = interp1(log_sw, y_idx, log(spv));
        yline(ax2, spv_idx, ':', 'Color', 'w', 'LineWidth', 0.6);
    end
    ylabel(ax2, '\sigma_{ph} (cents)');
    cb = colorbar(ax2, 'Position', [CBAR_L, ROW_B_Y, CBAR_W, ROW_B_H]);
    cb.Label.String = 'windowed similarity';
    title(ax2, '(b) Similarity heatmap over (offset, \sigma_{ph})');
    xlim(ax2, [offsets(1), offsets(end)]);

    % (c) Three overlaid profiles
    hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    cols = [0.12, 0.31, 0.72;
            0.17, 0.54, 0.24;
            0.76, 0.31, 0.03];
    legendStrs = cell(1, numel(sigma_ph_profiles));
    for i = 1:numel(sigma_ph_profiles)
        plot(ax3, offsets, prof(i, :), 'LineWidth', 1.7, 'Color', cols(i, :));
        legendStrs{i} = sprintf('\\sigma_{ph} = %g cents', sigma_ph_profiles(i));
    end
    for po = peak_offsets
        xline(ax3, po, '--', 'Color', [0.53, 0.53, 0.53], 'LineWidth', 0.7);
    end
    xlabel(ax3, 'Query time offset (s)');
    ylabel(ax3, 'Windowed similarity');
    legend(ax3, legendStrs, ...
        'Position', [LEG_L, ROW_C_Y + ROW_C_H - 0.095, LEG_W, 0.085], ...
        'FontSize', 9);
    title(ax3, '(c) Profiles at three representative \sigma_{ph}');
    xlim(ax3, [offsets(1), offsets(end)]);

    % Hide x-tick labels on top panels, link x-limits
    set(ax1, 'XTickLabel', []);
    set(ax2, 'XTickLabel', []);
    linkaxes([ax1, ax2, ax3], 'x');

    sgtitle(fig, suptitle_str, 'FontSize', 12);
end