%% demo_helixBlend.m
%  Helix blend: routing pitch through two groups of a MAET.
%
%  Demonstrates a multi-attribute expectation tensor pattern in which
%  the same pitch values are routed simultaneously through a periodic
%  pitch-class group and a linear register group. Sweeping the
%  register-group sigma while holding the pitch-class-group sigma fixed
%  morphs the similarity profile of a motif against a longer stream
%  from
%
%    * "matches every octave-displaced recurrence equally"  (large sigma_reg)
%    * through graded octave tolerance                      (medium)
%    * to "matches only the same-register recurrence"       (small sigma_reg).
%
%  Equivalence with Shepard's model. The factored Gaussian
%
%      exp(- d_pc(p1,p2)^2 / (2 sigma_pc^2))
%    * exp(-  (p1 - p2)^2  / (2 sigma_reg^2))
%
%  is equivalent to a Gaussian kernel of width sigma = sigma_pc on the
%  pitch-class-cum-register cylinder with stretch
%  h = sigma_pc/sigma_reg. Shepard's helix itself has no built-in
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
%            registers, with non-pitch-class-overlapping filler between
%            instances.
%
%    Part 2. Fugal texture in C minor (BWV 847-inspired, stylised; not
%            transcribed from the score). A six-note subject stated in
%            bass, alto, and soprano, with short counter-material
%            between entries.
%
%  Each part produces three stacked panels:
%    (a) the event stream,
%    (b) a similarity heatmap over (time offset, sigma_reg),
%    (c) three overlaid profile curves at representative sigma_reg
%        values.
%
%  Uses: buildExpTens, windowedSimilarity, convertPitch.

clear; clc; close all;

%% === User parameters ===

% -- Common --
SIG_PC           = 30;                               % pc group sigma (cents)
SIG_REG_SWEEP    = logspace(log10(100), log10(8000), 25);
SIG_REG_PROFILES = [200, 600, 3000];                 % three overlaid profiles
WIN_MIX          = 0.5;                              % rectangular x Gaussian

% -- Part 1 (synthetic) --
SIG_TIME_1       = 0.10;                             % sec
WIN_SIZE_TIME_1  = 6.0;                              % eff. sd in units of sigma_time
OFFSETS_1        = -0.5 : 0.02 : 12.5;

% -- Part 2 (fugal texture) --
SIG_TIME_2       = 0.08;
WIN_SIZE_TIME_2  = 12.0;
OFFSETS_2        = -0.5 : 0.02 : 8.5;

%% === Part 1: synthetic motif at four registers ===

fprintf('Part 1: synthetic motif at four registers.\n');
[ctx1_midi, ctx1_t, q1_midi, q1_t, motif_idx1, motif_cent1] = buildPart1Stream();
ctx1_cents = ctx1_midi * 100;
q1_cents   = q1_midi   * 100;

heat1 = sweepProfiles(q1_cents, q1_t, ctx1_cents, ctx1_t, ...
    SIG_PC, SIG_REG_SWEEP, SIG_TIME_1, WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1);
prof1 = sweepProfiles(q1_cents, q1_t, ctx1_cents, ctx1_t, ...
    SIG_PC, SIG_REG_PROFILES, SIG_TIME_1, WIN_SIZE_TIME_1, WIN_MIX, OFFSETS_1);
peak1 = motif_cent1 - mean(q1_t);

figure('Name', 'Helix blend: synthetic', 'Position', [80, 80, 980, 800]);
plotPart(gcf, 'Helix blend (synthetic): C-E-G at four registers', ...
    ctx1_midi, ctx1_t, motif_idx1, peak1, mean(q1_t), ...
    heat1, prof1, OFFSETS_1, SIG_REG_SWEEP, SIG_REG_PROFILES, ...
    'motif events (C-E-G)', 'filler events');

%% === Part 2: fugal texture (BWV 847-inspired, stylised) ===

fprintf('Part 2: fugal texture (BWV 847-inspired, stylised).\n');
[ctx2_midi, ctx2_t, q2_midi, q2_t, subj_idx2, subj_cent2] = buildPart2Stream();
ctx2_cents = ctx2_midi * 100;
q2_cents   = q2_midi   * 100;

heat2 = sweepProfiles(q2_cents, q2_t, ctx2_cents, ctx2_t, ...
    SIG_PC, SIG_REG_SWEEP, SIG_TIME_2, WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2);
prof2 = sweepProfiles(q2_cents, q2_t, ctx2_cents, ctx2_t, ...
    SIG_PC, SIG_REG_PROFILES, SIG_TIME_2, WIN_SIZE_TIME_2, WIN_MIX, OFFSETS_2);
peak2 = subj_cent2 - mean(q2_t);

figure('Name', 'Helix blend: fugal texture', 'Position', [120, 120, 980, 800]);
plotPart(gcf, ['Helix blend (BWV 847-inspired, stylised): ' ...
               'subject in bass, alto, soprano'], ...
    ctx2_midi, ctx2_t, subj_idx2, peak2, mean(q2_t), ...
    heat2, prof2, OFFSETS_2, SIG_REG_SWEEP, SIG_REG_PROFILES, ...
    'subject events', 'counter-material');

% =====================================================================
%  Local functions
% =====================================================================

function dens = build2Group(pitch_cents, time_sec, sigma_pc, sigma_reg, sigma_time)
%BUILD2GROUP  MAET density with pitch routed through two groups plus time.
%
%   Attributes: (pitch, pitch, time). Groups: (pc, reg, time), with pc
%   periodic at 1200 cents and the other two linear. All r = 1.
    p = pitch_cents(:).';
    t = time_sec(:).';
    dens = buildExpTens({p, p, t}, [], ...
        [sigma_pc, sigma_reg, sigma_time], ...
        [1, 1, 1], [], ...
        [false, false, false], ...
        [true,  false, false], ...
        [1200, 0, 0], ...
        'verbose', false);
end

function prof = sweepProfiles(q_cents, q_t, c_cents, c_t, ...
                              sigma_pc, sigma_reg_values, sigma_time, ...
                              win_size_time, win_mix, offsets)
%SWEEPPROFILES  Return a length(sigma_reg_values) x length(offsets) array
%   of cross-correlation windowed-similarity profiles.
    windowSpec = struct('size', [Inf, Inf, win_size_time], ...
                        'mix',  [0,   0,   win_mix]);
    % Offsets: pc and reg offsets = 0; sweep time offset
    N = numel(offsets);
    offsets_3d = [zeros(1, N); zeros(1, N); offsets(:).'];

    nS = numel(sigma_reg_values);
    prof = zeros(nS, N);
    for i = 1:nS
        sr = sigma_reg_values(i);
        dq = build2Group(q_cents, q_t, sigma_pc, sr, sigma_time);
        dc = build2Group(c_cents, c_t, sigma_pc, sr, sigma_time);
        prof(i, :) = windowedSimilarity(dq, dc, windowSpec, offsets_3d, ...
            'verbose', false);
    end
end

function [ctx_midi, ctx_t, q_midi, q_t, motif_idx, motif_cent] = buildPart1Stream()
%BUILDPART1STREAM  Three-note motif at four registers with non-overlapping filler.
    motif_midi_ref  = [60, 64, 67];        % C4 E4 G4
    filler_midi_ref = [62, 65, 69];        % D4 F4 A4 (disjoint pc's)
    regs_st         = [0, 12, -12, 24];
    dt              = 0.5;

    ctx_midi  = [];
    ctx_t     = [];
    motif_idx = [];
    t = 0;
    for k = 1:numel(regs_st)
        shift_st = regs_st(k);
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
    motif_cent = zeros(1, numel(regs_st));
    for k = 1:numel(regs_st)
        ii = motif_idx((k-1)*per_entry + 1 : k*per_entry);
        motif_cent(k) = mean(ctx_t(ii));
    end
end

function [ctx_midi, ctx_t, q_midi, q_t, subj_idx, subj_cent] = buildPart2Stream()
%BUILDPART2STREAM  Six-note subject at three registers, short counter-material between.
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
                  heat, prof, offsets, sigma_reg_sweep, sigma_reg_profiles, ...
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
    y_idx = 1:numel(sigma_reg_sweep);
    imagesc(ax2, offsets, y_idx, heat);
    caxis(ax2, [0, cmax]);
    set(ax2, 'YDir', 'normal');
    colormap(ax2, parula);

    log_sw = log(sigma_reg_sweep);
    cand = [100, 200, 500, 1000, 2000, 5000];
    ytick_vals = cand(cand >= min(sigma_reg_sweep) & cand <= max(sigma_reg_sweep));
    ytick_pos  = interp1(log_sw, y_idx, log(ytick_vals));
    set(ax2, 'YTick', ytick_pos, ...
        'YTickLabel', arrayfun(@(v) sprintf('%g', v), ytick_vals, ...
                               'UniformOutput', false));

    hold(ax2, 'on');
    for po = peak_offsets
        xline(ax2, po, '--', 'Color', 'w', 'LineWidth', 0.7);
    end
    for spv = sigma_reg_profiles
        spv_idx = interp1(log_sw, y_idx, log(spv));
        yline(ax2, spv_idx, ':', 'Color', 'w', 'LineWidth', 0.6);
    end
    ylabel(ax2, '\sigma_{reg} (cents)');
    cb = colorbar(ax2, 'Position', [CBAR_L, ROW_B_Y, CBAR_W, ROW_B_H]);
    cb.Label.String = 'windowed similarity';
    title(ax2, '(b) Similarity heatmap over (offset, \sigma_{reg})');
    xlim(ax2, [offsets(1), offsets(end)]);

    % (c) Three overlaid profiles
    hold(ax3, 'on'); grid(ax3, 'on'); box(ax3, 'on');
    cols = [0.12, 0.31, 0.72;
            0.17, 0.54, 0.24;
            0.76, 0.31, 0.03];
    legendStrs = cell(1, numel(sigma_reg_profiles));
    for i = 1:numel(sigma_reg_profiles)
        plot(ax3, offsets, prof(i, :), 'LineWidth', 1.7, 'Color', cols(i, :));
        legendStrs{i} = sprintf('\\sigma_{reg} = %g cents', sigma_reg_profiles(i));
    end
    for po = peak_offsets
        xline(ax3, po, '--', 'Color', [0.53, 0.53, 0.53], 'LineWidth', 0.7);
    end
    xlabel(ax3, 'Query time offset (s)');
    ylabel(ax3, 'Cosine similarity');
    legend(ax3, legendStrs, ...
        'Position', [LEG_L, ROW_C_Y + ROW_C_H - 0.095, LEG_W, 0.085], ...
        'FontSize', 9);
    title(ax3, '(c) Profiles at three representative \sigma_{reg}');
    xlim(ax3, [offsets(1), offsets(end)]);

    % Hide x-tick labels on top panels, link x-limits
    set(ax1, 'XTickLabel', []);
    set(ax2, 'XTickLabel', []);
    linkaxes([ax1, ax2, ax3], 'x');

    sgtitle(fig, suptitle_str, 'FontSize', 12);
end