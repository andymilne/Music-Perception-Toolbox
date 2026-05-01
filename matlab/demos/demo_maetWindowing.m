%% demo_maetWindowing.m
%  MAET post-tensor windowing over time and pitch.
%
%  Pedagogical demonstration of four time-windowed pipelines for locating
%  a three-event motif in a longer context, plus one pitch-windowed
%  pipeline that interrogates harmonic and subharmonic relationships at
%  a fixed time offset:
%
%    raw               Absolute per-event pitch. Only literal-pitch
%                      repetitions of the motif register.
%
%    differenced       Inter-event pitch intervals (via
%                      differenceEvents). Transposition-invariant;
%                      time order matters.
%
%    spectrum          Spectrum-enriched pitch (via addSpectra, harmonic
%                      partials). Literal matches plus partial-shared
%                      transpositions (P4, P5).
%
%    spectrum + decay  Spectrum enrichment with an exponential decay
%                      applied to the context's per-event weights (query
%                      weights left uniform), modelling memory fade for
%                      older events. Late-motif matches gain relative
%                      weight at the expense of early literal matches.
%
%    pitch @ M4        Pitch-windowed scan at the time offset where the
%                      query lands on M4 (A-B-C, the P5 transposition of
%                      the D-E-F query). Both operands are spectrum-
%                      enriched, so peaks in the pitch profile reveal
%                      the intervallic structure of partial-sharing: the
%                      principal peak sits at +700 cents (the P5 up to
%                      M4's centroid) with smaller secondary peaks at
%                      octave and subharmonic offsets.
%
%    pitch x time      2-D version of the pitch row: sweep both time
%                      and pitch offsets simultaneously. Shown in a
%                      separate figure for a harmonic query (reference)
%                      and for a stretched-partial query (inharmonic
%                      comparison).
%
%  For the first four rows, sweeps use the cross-correlation offset API
%  with the time group windowed and the pitch group unwindowed. Row 5
%  windows both groups simultaneously, fixing the time offset at the M4
%  centroid and sweeping pitch. Offsets are measured from each query's
%  unweighted centroid to the window centre (the default reference used
%  by windowedSimilarity), so offset zero sits on the query's own geometric
%  centre.
%
%  Reference-point choice. windowedSimilarity offers two reference-point
%  options for the cross-correlation: the default (unweighted column
%  mean of the query's tuple centres) and a user-supplied fixed
%  reference. This demo uses the default throughout. Which method is
%  appropriate depends on how queries vary between runs: see
%  USER_GUIDE.md §3.1 "Post-tensor windowing" for a summary, and the
%  demo demo_windowingReference for a full analysis across slot-count,
%  slot-value, and slot-weight sweeps of both harmonic and non-harmonic
%  queries.
%
%  Uses: buildExpTens, evalExpTens, cosSimExpTens, windowedSimilarity,
%        differenceEvents, addSpectra, convertPitch.

%% === User-tweakable parameters ===

% Perceptual uncertainties
PITCH_SIGMA_CENTS = 12.0;
TIME_SIGMA_SEC    = 0.030;

% Window shape
WIN_SIZE_TIME = 25;             % time window effective sd, in TIME_SIGMA_SEC units  (~= 0.75 s)
WIN_MIX       = 0.5;

% Geometry
PITCH_PERIODIC = false;
TIME_PERIODIC  = false;
BAR_SEC        = 2.0;

% Visualisation sweep parameters (rows 1-4). Alignment rule as for
% TIME_STEP_2D (see 2-D section below): 0.01 s is a subdivision of
% 0.25 s, and < TIME_SIGMA_SEC/2, so it aligns with every event and
% resolves peak shape.
TIME_STEP_1D = 0.01;
OFFSET_LO    = -0.8;
OFFSET_HI    =  7.8;

% Spectrum enrichment (rows 3, 4, 5)
N_PARTIALS  = 12;
ROLLOFF_EXP = 0.5;

% Context exponential decay (row 4)
CONTEXT_DECAY_RATE = 0.3;

% Pitch windowing at M4 (row 5)
M4_TIME_OFFSET      = 6.5;
PITCH_OFFSET_RANGE  = 3600.0;
PITCH_STEP_1D       = 5.0;       % cents per sample -- subdivision of 100 c, < PITCH_SIGMA_CENTS/2
WIN_SIZE_PITCH_EFF  = 400.0;

% 2-D pitch x time windowing (second figure)
% To line up the grid with the context event coordinates, TIME_STEP_2D
% must be a subdivision of 0.25 s (the GCD of event times in this
% example) and PITCH_STEP_2D a subdivision of 100 cents (GCD of event
% pitches, set by the chromatic scale). To resolve peak shape as well
% as location, steps smaller than half the relevant sigma are needed
% (e.g. 0.01 s < TIME_SIGMA_SEC/2, 5 cents < PITCH_SIGMA_CENTS/2);
% finer grids are slower to compute.
PITCH_STEP_2D = 100.0;
TIME_STEP_2D  = 0.05;

% Inharmonic query spectrum (second figure row B)
STRETCH_BETA = 1.05;

% Density-panel grids
DENS_DP_SPEC  = 4.0;
DENS_DT_SPEC  = 0.03;
DENS_DP_PLAIN = 20.0;
DENS_DT_PLAIN = 0.05;

%% === Musical example ===

events_tm = [ ...
    0.00, 62;  0.50, 64;  1.00, 65;  1.50, 64;  2.00, 64;  2.50, 65;  ...
    3.00, 67;  3.25, 65;  3.50, 64;  4.00, 67;  4.50, 69;  5.00, 70;  ...
    6.00, 72;  6.25, 70;  6.50, 69;  7.00, 71;  7.50, 72;  8.00, 74];

context_time  = events_tm(:, 1).';
context_midi  = events_tm(:, 2).';
context_cents = convertPitch(context_midi, 'midi', 'cents');

query_time  = context_time(1:3);
query_cents = context_cents(1:3);

motif_offsets = [0.0, 2.0, 4.0, 6.5];
motif_labels  = {'M1', 'M2', 'M3', 'M4'};
motif_roots   = {'D',  'E',  'G',  'A'};

if PITCH_PERIODIC, pitch_period = 1200.0; else, pitch_period = 0.0; end
if TIME_PERIODIC,  time_period  = BAR_SEC; else, time_period  = 0.0; end

%% === Build densities ===

fprintf('Building densities ...\n');

context_decay_w = exp(-CONTEXT_DECAY_RATE * (max(context_time) - context_time));

% Row 1: raw
dens_ctx_raw = buildExpTens({context_cents, context_time}, [], ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);
dens_q_raw   = buildExpTens({query_cents, query_time}, [], ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);

% Row 2: differenced
[p_diff_ctx, w_diff_ctx] = differenceEvents({context_cents, context_time}, ...
    [], [], [1, 0], [pitch_period, time_period]);
[p_diff_q,   w_diff_q]   = differenceEvents({query_cents,   query_time}, ...
    [], [], [1, 0], [pitch_period, time_period]);

dens_ctx_diff = buildExpTens(p_diff_ctx, w_diff_ctx, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);
dens_q_diff   = buildExpTens(p_diff_q, w_diff_q, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);

% Row 3: spectrum-enriched (all ones)
[p_flat_ctx, w_flat_ctx] = addSpectra(context_cents.', [], ...
    'harmonic', N_PARTIALS, 'powerlaw', ROLLOFF_EXP);
[p_flat_q,   w_flat_q]   = addSpectra(query_cents.', [], ...
    'harmonic', N_PARTIALS, 'powerlaw', ROLLOFF_EXP);

n_ctx = numel(context_cents);
n_q   = numel(query_cents);
p_spec_ctx = reshape(p_flat_ctx, n_ctx, N_PARTIALS).';
w_spec_ctx = reshape(w_flat_ctx, n_ctx, N_PARTIALS).';
p_spec_q   = reshape(p_flat_q,   n_q,   N_PARTIALS).';
w_spec_q   = reshape(w_flat_q,   n_q,   N_PARTIALS).';

dens_ctx_spec = buildExpTens({p_spec_ctx, context_time}, {w_spec_ctx, []}, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);
dens_q_spec   = buildExpTens({p_spec_q, query_time}, {w_spec_q, []}, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);

% Row 4: spectrum + context decay
[p_flat_ctxd, w_flat_ctxd] = addSpectra(context_cents.', context_decay_w.', ...
    'harmonic', N_PARTIALS, 'powerlaw', ROLLOFF_EXP);
p_specd_ctx = reshape(p_flat_ctxd, n_ctx, N_PARTIALS).';
w_specd_ctx = reshape(w_flat_ctxd, n_ctx, N_PARTIALS).';

dens_ctx_spec_d = buildExpTens({p_specd_ctx, context_time}, {w_specd_ctx, []}, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);
dens_q_spec_d = dens_q_spec;

% Second figure, row B: stretched-partial query
[p_flat_qs, w_flat_qs] = addSpectra(query_cents.', [], ...
    'stretched', N_PARTIALS, STRETCH_BETA, 'powerlaw', ROLLOFF_EXP);
p_qstretch = reshape(p_flat_qs, n_q, N_PARTIALS).';
w_qstretch = reshape(w_flat_qs, n_q, N_PARTIALS).';

dens_q_stretch = buildExpTens({p_qstretch, query_time}, {w_qstretch, []}, ...
    [PITCH_SIGMA_CENTS, TIME_SIGMA_SEC], [1, 1], [], ...
    [false, false], [PITCH_PERIODIC, TIME_PERIODIC], ...
    [pitch_period, time_period], 'verbose', false);

%% === Time-window sweeps (rows 1-4) ===

spec_time_only = struct('size', [Inf, WIN_SIZE_TIME], 'mix', [0.0, WIN_MIX]);

i_lo = ceil(OFFSET_LO / TIME_STEP_1D);
i_hi = floor(OFFSET_HI / TIME_STEP_1D);
offset_sweep = (i_lo:i_hi) * TIME_STEP_1D;
N_SWEEP = numel(offset_sweep);
offsets_2d = [zeros(1, N_SWEEP); offset_sweep];

fprintf('Computing windowed similarity profiles (rows 1-4) ...\n');

profile_raw    = windowedSimilarity(dens_q_raw,    dens_ctx_raw,    spec_time_only, offsets_2d, 'verbose', false);
profile_diff   = windowedSimilarity(dens_q_diff,   dens_ctx_diff,   spec_time_only, offsets_2d, 'verbose', false);
profile_spec   = windowedSimilarity(dens_q_spec,   dens_ctx_spec,   spec_time_only, offsets_2d, 'verbose', false);
profile_spec_d = windowedSimilarity(dens_q_spec_d, dens_ctx_spec_d, spec_time_only, offsets_2d, 'verbose', false);

%% === Pitch sweep at M4 (row 5) ===

win_size_pitch = WIN_SIZE_PITCH_EFF / PITCH_SIGMA_CENTS;
spec_both = struct('size', [win_size_pitch, WIN_SIZE_TIME], ...
                   'mix',  [WIN_MIX, WIN_MIX]);

i_plo = ceil(-PITCH_OFFSET_RANGE / PITCH_STEP_1D);
i_phi = floor(PITCH_OFFSET_RANGE / PITCH_STEP_1D);
pitch_offset_sweep = (i_plo:i_phi) * PITCH_STEP_1D;
N_PITCH_SWEEP = numel(pitch_offset_sweep);
offsets_pitch_2d = [pitch_offset_sweep; M4_TIME_OFFSET * ones(1, N_PITCH_SWEEP)];

fprintf('Computing pitch-windowed profile at M4 ...\n');
profile_pitch = windowedSimilarity(dens_q_spec, dens_ctx_spec, spec_both, ...
    offsets_pitch_2d, 'verbose', false);

%% === 2-D pitch x time sweeps (second figure) ===

i_p2_lo = ceil(-PITCH_OFFSET_RANGE / PITCH_STEP_2D);
i_p2_hi = floor(PITCH_OFFSET_RANGE / PITCH_STEP_2D);
pitch_offset_2d = (i_p2_lo:i_p2_hi) * PITCH_STEP_2D;
N_PITCH_2D = numel(pitch_offset_2d);

i_t2_lo = ceil(OFFSET_LO / TIME_STEP_2D);
i_t2_hi = floor(OFFSET_HI / TIME_STEP_2D);
time_offset_2d = (i_t2_lo:i_t2_hi) * TIME_STEP_2D;
N_TIME_2D = numel(time_offset_2d);

[PP_2D, TT_2D] = meshgrid(pitch_offset_2d, time_offset_2d);
offsets_pt_2d = [PP_2D(:).'; TT_2D(:).'];

fprintf('Computing 2-D pitch x time sweep (%d x %d = %d points, harmonic query) ...\n', ...
    N_PITCH_2D, N_TIME_2D, size(offsets_pt_2d, 2));
tic;
profile_pt_flat = windowedSimilarity(dens_q_spec, dens_ctx_spec, spec_both, ...
    offsets_pt_2d, 'verbose', false);
fprintf('  done in %.1f s\n', toc);
profile_pt = reshape(profile_pt_flat, N_TIME_2D, N_PITCH_2D);

fprintf('Computing 2-D pitch x time sweep (query partials stretched, beta = %g) ...\n', ...
    STRETCH_BETA);
tic;
profile_pt_stretch_flat = windowedSimilarity(dens_q_stretch, dens_ctx_spec, ...
    spec_both, offsets_pt_2d, 'verbose', false);
fprintf('  done in %.1f s\n', toc);
profile_pt_stretch = reshape(profile_pt_stretch_flat, N_TIME_2D, N_PITCH_2D);

%% === Main figure (5 rows) ===

fprintf('Drawing main figure ...\n');

fig = figure('Units', 'pixels', 'Position', [100, 100, 1400, 1600], 'Color', 'w');
set(fig, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
tl = tiledlayout(fig, 5, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

q_diff_t_all   = p_diff_q{2}(:).';
ctx_diff_p_all = p_diff_ctx{1}(:).';
q_diff_p_all   = p_diff_q{1}(:).';

row_specs = { ...
    struct('dens_ctx', dens_ctx_raw,    'dens_q', dens_q_raw,    ...
           'p_ctx_range', context_cents, 'p_q_range', query_cents, ...
           'p_label', 'pitch (cents)', 'has_partials', false, ...
           't_q', query_time, ...
           'kind', 'time', 'data', profile_raw, 'with_roots', false, ...
           'extra_title', '', 'row_label', sprintf('raw\n(per-event pitch)')); ...
    struct('dens_ctx', dens_ctx_diff,   'dens_q', dens_q_diff,   ...
           'p_ctx_range', ctx_diff_p_all, 'p_q_range', q_diff_p_all, ...
           'p_label', 'pitch interval (cents)', 'has_partials', false, ...
           't_q', q_diff_t_all, ...
           'kind', 'time', 'data', profile_diff, 'with_roots', false, ...
           'extra_title', '', 'row_label', sprintf('differenced\n(inter-event intervals)')); ...
    struct('dens_ctx', dens_ctx_spec,   'dens_q', dens_q_spec,   ...
           'p_ctx_range', p_spec_ctx(:).', 'p_q_range', p_spec_q(:).', ...
           'p_label', 'pitch (cents)', 'has_partials', true, ...
           't_q', query_time, ...
           'kind', 'time', 'data', profile_spec, 'with_roots', false, ...
           'extra_title', '', 'row_label', sprintf('spectrum\n(%d partials, \\rho = %g)', N_PARTIALS, ROLLOFF_EXP)); ...
    struct('dens_ctx', dens_ctx_spec_d, 'dens_q', dens_q_spec_d, ...
           'p_ctx_range', p_specd_ctx(:).', 'p_q_range', p_spec_q(:).', ...
           'p_label', 'pitch (cents)', 'has_partials', true, ...
           't_q', query_time, ...
           'kind', 'time', 'data', profile_spec_d, 'with_roots', true, ...
           'extra_title', '', 'row_label', sprintf('spectrum + decay\n(context rate = %g/s)', CONTEXT_DECAY_RATE)); ...
    struct('dens_ctx', dens_ctx_spec,   'dens_q', dens_q_spec,   ...
           'p_ctx_range', p_spec_ctx(:).', 'p_q_range', p_spec_q(:).', ...
           'p_label', 'pitch (cents)', 'has_partials', true, ...
           't_q', query_time, ...
           'kind', 'pitch', 'data', profile_pitch, 'with_roots', false, ...
           'extra_title', '  (pitch sweep at M4)', 'row_label', sprintf('pitch @ M4\n(win size (p) \\approx %g c)', WIN_SIZE_PITCH_EFF)); ...
};

for row_idx = 1:5
    rs = row_specs{row_idx};
    localMakeRow(tl, row_idx, rs, context_time, ...
        DENS_DP_SPEC, DENS_DT_SPEC, DENS_DP_PLAIN, DENS_DT_PLAIN, ...
        offset_sweep, pitch_offset_sweep, ...
        motif_offsets, motif_labels, motif_roots, ...
        M4_TIME_OFFSET);
end

title(tl, sprintf(['MAET windowing over time and pitch  ' ...
    '[pitch \\sigma = %g c, time \\sigma = %g ms, ' ...
    'win size (t) = %d\\sigma, mix = %g]'], ...
    PITCH_SIGMA_CENTS, TIME_SIGMA_SEC * 1000, WIN_SIZE_TIME, WIN_MIX), ...
    'FontSize', 12, 'Color', 'k');

% Row labels on the left margin, rendered as rotated text on a
% full-figure invisible axes. Row y-centres are computed from the
% outer tiledlayout's geometry (equal-height rows under a small top
% margin for the suptitle), so the labels are always at the correct
% position regardless of MATLAB's layout timing.
TOP_MARGIN_MAIN = 0.05;
BOT_MARGIN_MAIN = 0.02;
N_ROWS_MAIN = 5;
row_h_main = (1 - TOP_MARGIN_MAIN - BOT_MARGIN_MAIN) / N_ROWS_MAIN;

ax_labels = axes(fig, 'Position', [0 0 1 1], 'Visible', 'off', ...
                 'XLim', [0 1], 'YLim', [0 1], ...
                 'HitTest', 'off', 'PickableParts', 'none');
for row_idx = 1:N_ROWS_MAIN
    rs = row_specs{row_idx};
    y_centre = 1 - TOP_MARGIN_MAIN - (row_idx - 0.5) * row_h_main;
    text(ax_labels, 0.012, y_centre, rs.row_label, ...
         'Rotation', 90, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'FontSize', 11, 'FontWeight', 'bold', ...
         'Color', 'k', 'Interpreter', 'tex');
end
uistack(ax_labels, 'top');

%% === Second figure: 2-D pitch x time, harmonic vs stretched query ===

fprintf('Drawing pitch x time figure ...\n');

fig2 = figure('Units', 'pixels', 'Position', [100, 100, 1400, 1100], 'Color', 'w');
set(fig2, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
tl2 = tiledlayout(fig2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

pt_specs = { ...
    struct('dens_ctx', dens_ctx_spec, 'dens_q', dens_q_spec, ...
           'p_ctx_range', p_spec_ctx(:).', 'p_q_range', p_spec_q(:).', ...
           'profile_pt', profile_pt, ...
           'row_label', sprintf('harmonic query\n(\\beta = 1.0)')); ...
    struct('dens_ctx', dens_ctx_spec, 'dens_q', dens_q_stretch, ...
           'p_ctx_range', p_spec_ctx(:).', 'p_q_range', p_qstretch(:).', ...
           'profile_pt', profile_pt_stretch, ...
           'row_label', sprintf('stretched query\n(\\beta = %g)', STRETCH_BETA)); ...
};

for row_idx = 1:2
    ps = pt_specs{row_idx};
    localMakePtRow(tl2, row_idx, ps, context_time, query_time, ...
        DENS_DP_SPEC, DENS_DT_SPEC, ...
        pitch_offset_2d, time_offset_2d, ...
        motif_offsets, motif_labels);
end

title(tl2, sprintf(['2-D pitch \\times time similarity: ' ...
    'harmonic vs stretched query spectrum  ' ...
    '(context harmonic, %d partials, \\rho = %g)'], ...
    N_PARTIALS, ROLLOFF_EXP), 'FontSize', 12, 'Color', 'k');

% Row labels on the left margin of fig2 (same approach as fig).
TOP_MARGIN_PT = 0.07;
BOT_MARGIN_PT = 0.03;
N_ROWS_PT = 2;
row_h_pt = (1 - TOP_MARGIN_PT - BOT_MARGIN_PT) / N_ROWS_PT;

ax_labels2 = axes(fig2, 'Position', [0 0 1 1], 'Visible', 'off', ...
                  'XLim', [0 1], 'YLim', [0 1], ...
                  'HitTest', 'off', 'PickableParts', 'none');
for row_idx = 1:N_ROWS_PT
    ps = pt_specs{row_idx};
    y_centre = 1 - TOP_MARGIN_PT - (row_idx - 0.5) * row_h_pt;
    text(ax_labels2, 0.012, y_centre, ps.row_label, ...
         'Rotation', 90, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'FontSize', 11, 'FontWeight', 'bold', ...
         'Color', 'k', 'Interpreter', 'tex');
end
uistack(ax_labels2, 'top');

%% === Peak summary ===

fprintf('\nTime-windowed peak similarity at motif offsets:\n');
fprintf('  %4s  %4s  %7s  %7s  %7s  %7s  %7s\n', ...
        'name', 'root', 'offset', 'raw', 'diff', 'spec', 'spec+d');
for i = 1:numel(motif_labels)
    off = motif_offsets(i);
    fprintf('  %4s  %4s  %7.2f  %7.3f  %7.3f  %7.3f  %7.3f\n', ...
        motif_labels{i}, motif_roots{i}, off, ...
        localPeakNear(offset_sweep, profile_raw,    off, 0.3), ...
        localPeakNear(offset_sweep, profile_diff,   off, 0.3), ...
        localPeakNear(offset_sweep, profile_spec,   off, 0.3), ...
        localPeakNear(offset_sweep, profile_spec_d, off, 0.3));
end

fprintf('\nPitch-windowed peak similarity at reference intervals (t = %g s, M4):\n', ...
    M4_TIME_OFFSET);
fprintf('  %10s  %7s  %7s\n', 'interval', 'cents', 'sim');
ref_labels = {'-2 oct', '-1 oct', '-P5', '-P4', 'unison', '+P4', '+P5 (M4)', '+1 oct', '+2 oct'};
ref_cents  = [-2400, -1200, -700, -500, 0, 500, 700, 1200, 2400];
for i = 1:numel(ref_labels)
    fprintf('  %10s  %+7d  %7.3f\n', ref_labels{i}, ref_cents(i), ...
        localPeakNear(pitch_offset_sweep, profile_pitch, ref_cents(i), 60.0));
end

[max_val, i_max] = max(profile_pitch);
fprintf('\n  Global max at offset = %+.1f cents  (sim = %.3f)\n', ...
    pitch_offset_sweep(i_max), max_val);

fprintf('\nDone.\n');

%% === Local helpers ===

function localMakeRow(tl, row_idx, rs, context_time, ...
    dp_spec, dt_spec, dp_plain, dt_plain, ...
    offset_sweep, pitch_offset_sweep, ...
    motif_offsets, motif_labels, motif_roots, M4_TIME_OFFSET)
% Build one row: context density | query density | profile.
%
% The density-block tile is split by a nested tiledlayout so that the
% two panels share the same pitch y-range and the same seconds-per-inch
% scaling. After layout, panel widths are adjusted to match time-extent
% ratios.

    p_vals = [rs.p_ctx_range(:); rs.p_q_range(:)];
    p_lo = min(p_vals) - 100.0;
    p_hi = max(p_vals) + 100.0;

    t_lo_c = -0.3;
    t_hi_c = max(context_time) + 0.3;
    t_lo_q = min(rs.t_q) - 0.3;
    t_hi_q = max(rs.t_q) + 0.3;

    if rs.has_partials
        dp = dp_spec;   dt = dt_spec;
    else
        dp = dp_plain;  dt = dt_plain;
    end

    p_grid   = (ceil(p_lo/dp)   : floor(p_hi/dp))   * dp;
    t_grid_c = (ceil(t_lo_c/dt) : floor(t_hi_c/dt)) * dt;
    t_grid_q = (ceil(t_lo_q/dt) : floor(t_hi_q/dt)) * dt;

    Z_c = localEvalDensity(rs.dens_ctx, p_grid, t_grid_c);
    Z_q = localEvalDensity(rs.dens_q,   p_grid, t_grid_q);
    vmax = max(max(Z_c(:)), max(Z_q(:)));

    ctx_extent = t_hi_c - t_lo_c;
    q_extent   = t_hi_q - t_lo_q;

    % Layout policy for the density block:
    %   - Context panel: fixed width, always 60% of the row.
    %   - Gap: whitespace between context and query.
    %   - Query panel: scaled so seconds-per-axis-unit matches the
    %     context panel.
    %   - Trailing whitespace to the right of the query fills the row.
    % Implemented by nesting a 75-column tiledlayout inside the row's
    % left tile and placing axes at specific column spans.
    N_INNER_COLS = 75;
    CTX_SPAN   = 60;
    GAP_SPAN   = 2;
    q_span     = max(1, round(CTX_SPAN * q_extent / ctx_extent));
    q_start    = CTX_SPAN + GAP_SPAN + 1;
    q_span     = min(q_span, N_INNER_COLS - q_start + 1);

    inner_tl = tiledlayout(tl, 1, N_INNER_COLS, ...
        'TileSpacing', 'none', 'Padding', 'none');
    inner_tl.Layout.Tile = (row_idx - 1) * 2 + 1;

    ax_ctx = nexttile(inner_tl, 1,       [1, CTX_SPAN]);
    ax_q   = nexttile(inner_tl, q_start, [1, q_span]);

    imagesc(ax_ctx, t_grid_c, p_grid, Z_c);
    set(ax_ctx, 'YDir', 'normal');
    colormap(ax_ctx, parula);
    caxis(ax_ctx, [0, vmax]);
    xlabel(ax_ctx, 'time (s)');
    ylabel(ax_ctx, rs.p_label);
    title(ax_ctx, sprintf('context density%s', rs.extra_title), 'Color', 'k');
    set(ax_ctx, 'Tag', sprintf('row%d_ctx', row_idx));
    xlim(ax_ctx, [t_lo_c, t_hi_c]);
    ylim(ax_ctx, [p_lo, p_hi]);

    imagesc(ax_q, t_grid_q, p_grid, Z_q);
    set(ax_q, 'YDir', 'normal');
    colormap(ax_q, parula);
    caxis(ax_q, [0, vmax]);
    xlabel(ax_q, 'time (s)');
    set(ax_q, 'YTickLabel', {});
    title(ax_q, 'query', 'Color', 'k');
    set(ax_q, 'Tag', sprintf('row%d_q', row_idx));
    xlim(ax_q, [t_lo_q, t_hi_q]);
    ylim(ax_q, [p_lo, p_hi]);

    ax_pr = nexttile(tl, (row_idx - 1) * 2 + 2);
    switch rs.kind
        case 'time'
            localPlotTimeProfile(ax_pr, offset_sweep, rs.data, ...
                motif_offsets, motif_labels, motif_roots, rs.with_roots);
        case 'pitch'
            localPlotPitchProfile(ax_pr, pitch_offset_sweep, rs.data, ...
                M4_TIME_OFFSET);
    end
end


function localMakePtRow(tl, row_idx, ps, context_time, query_time, ...
    dp_spec, dt_spec, pitch_offset_2d, time_offset_2d, ...
    motif_offsets, motif_labels)
% Second figure row: context density | query density | 2-D heatmap.

    p_vals = [ps.p_ctx_range(:); ps.p_q_range(:)];
    p_lo = min(p_vals) - 100.0;
    p_hi = max(p_vals) + 100.0;

    t_lo_c = -0.3;  t_hi_c = max(context_time) + 0.3;
    t_lo_q = min(query_time) - 0.3;  t_hi_q = max(query_time) + 0.3;

    dp = dp_spec;  dt = dt_spec;
    p_grid   = (ceil(p_lo/dp)   : floor(p_hi/dp))   * dp;
    t_grid_c = (ceil(t_lo_c/dt) : floor(t_hi_c/dt)) * dt;
    t_grid_q = (ceil(t_lo_q/dt) : floor(t_hi_q/dt)) * dt;

    Z_c = localEvalDensity(ps.dens_ctx, p_grid, t_grid_c);
    Z_q = localEvalDensity(ps.dens_q,   p_grid, t_grid_q);
    vmax = max(max(Z_c(:)), max(Z_q(:)));

    ctx_extent = t_hi_c - t_lo_c;
    q_extent   = t_hi_q - t_lo_q;

    N_INNER_COLS = 75;
    CTX_SPAN   = 60;
    GAP_SPAN   = 2;
    q_span     = max(1, round(CTX_SPAN * q_extent / ctx_extent));
    q_start    = CTX_SPAN + GAP_SPAN + 1;
    q_span     = min(q_span, N_INNER_COLS - q_start + 1);

    inner_tl = tiledlayout(tl, 1, N_INNER_COLS, ...
        'TileSpacing', 'none', 'Padding', 'none');
    inner_tl.Layout.Tile = (row_idx - 1) * 2 + 1;

    ax_ctx = nexttile(inner_tl, 1,       [1, CTX_SPAN]);
    ax_q   = nexttile(inner_tl, q_start, [1, q_span]);

    imagesc(ax_ctx, t_grid_c, p_grid, Z_c);
    set(ax_ctx, 'YDir', 'normal');
    colormap(ax_ctx, parula);
    caxis(ax_ctx, [0, vmax]);
    xlabel(ax_ctx, 'time (s)'); ylabel(ax_ctx, 'pitch (cents)');
    title(ax_ctx, 'context density', 'Color', 'k');
    set(ax_ctx, 'Tag', sprintf('pt_row%d_ctx', row_idx));
    xlim(ax_ctx, [t_lo_c, t_hi_c]);
    ylim(ax_ctx, [p_lo, p_hi]);

    imagesc(ax_q, t_grid_q, p_grid, Z_q);
    set(ax_q, 'YDir', 'normal');
    colormap(ax_q, parula);
    caxis(ax_q, [0, vmax]);
    xlabel(ax_q, 'time (s)'); set(ax_q, 'YTickLabel', {});
    title(ax_q, 'query', 'Color', 'k');
    xlim(ax_q, [t_lo_q, t_hi_q]);
    ylim(ax_q, [p_lo, p_hi]);

    ax_pr = nexttile(tl, (row_idx - 1) * 2 + 2);
    imagesc(ax_pr, time_offset_2d, pitch_offset_2d, ps.profile_pt.');
    set(ax_pr, 'YDir', 'normal');
    colormap(ax_pr, parula);
    xlabel(ax_pr, 'time offset from query centroid (s)');
    ylabel(ax_pr, 'pitch offset (cents)');
    title(ax_pr, '2-D pitch \times time similarity', 'Color', 'k');

    tick_span = 600;
    tick_max = floor(pitch_offset_2d(end) / tick_span) * tick_span;
    set(ax_pr, 'YTick', -tick_max:tick_span:tick_max);

    xlim(ax_pr, [time_offset_2d(1), time_offset_2d(end)]);
    ylim(ax_pr, [pitch_offset_2d(1), pitch_offset_2d(end)]);
    box(ax_pr, 'off');

    ylims = ylim(ax_pr);
    for i = 1:numel(motif_offsets)
        text(ax_pr, motif_offsets(i), ylims(2), motif_labels{i}, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'Color', 'w');
    end
end


function Z = localEvalDensity(dens, p_grid, t_grid)
% Evaluate a MAET density on a pitch x time grid. Returns Z with shape
% [numel(p_grid), numel(t_grid)] so imagesc(t_grid, p_grid, Z) renders
% pitch on the y-axis and time on the x-axis.
    [PP, TT] = meshgrid(p_grid, t_grid);
    X = [PP(:).'; TT(:).'];
    vals = evalExpTens(dens, X, 'verbose', false);
    Z = reshape(vals, numel(t_grid), numel(p_grid));  % [t, p]
    Z = Z.';                                          % [p, t]
end


function localPlotTimeProfile(ax, offset_sweep, profile, ...
    motif_offsets, motif_labels, motif_roots, with_roots)
    ymax = max(profile) * 1.15;
    plot(ax, offset_sweep, profile, '-', 'LineWidth', 1.6, ...
         'Color', [0, 0.447, 0.741]);
    hold(ax, 'on');

    for i = 1:numel(motif_offsets)
        off = motif_offsets(i);
        xline(ax, off, 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);
        if with_roots
            lbl = sprintf('%s %s', motif_labels{i}, motif_roots{i});
        else
            lbl = motif_labels{i};
        end
        text(ax, off, ymax, lbl, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'Color', [0.25, 0.25, 0.25]);
    end
    hold(ax, 'off');

    xlabel(ax, 'window offset from query centroid (s)');
    ylabel(ax, 'windowed similarity');
    title(ax, 'similarity profile', 'Color', 'k');
    xlim(ax, [offset_sweep(1), offset_sweep(end)]);
    ylim(ax, [0, ymax]);
    box(ax, 'off');
end


function localPlotPitchProfile(ax, pitch_offset_sweep, profile, M4_TIME_OFFSET)
    ymax = max(profile) * 1.15;
    plot(ax, pitch_offset_sweep, profile, '-', 'LineWidth', 1.6, ...
         'Color', [0, 0.447, 0.741]);
    hold(ax, 'on');

    tick_span = 600;
    tick_max = floor(pitch_offset_sweep(end) / tick_span) * tick_span;
    set(ax, 'XTick', -tick_max:tick_span:tick_max);

    ref_cents  = [-2400, -1200, -700, 0, 700, 1200, 2400];
    ref_labels = {'-2 oct', '-1 oct', '-P5', '0', '+P5', '+1 oct', '+2 oct'};
    for i = 1:numel(ref_cents)
        xline(ax, ref_cents(i), 'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8);
        text(ax, ref_cents(i), ymax, ref_labels{i}, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'Color', [0.25, 0.25, 0.25]);
    end
    hold(ax, 'off');

    xlabel(ax, 'window offset from query centroid (cents)');
    ylabel(ax, 'windowed similarity');
    title(ax, sprintf('pitch-windowed profile at t = %g s (M4)', M4_TIME_OFFSET), 'Color', 'k');
    xlim(ax, [pitch_offset_sweep(1), pitch_offset_sweep(end)]);
    ylim(ax, [0, ymax]);
    box(ax, 'off');
end


function peak = localPeakNear(offs, profile, target, radius)
    mask = abs(offs - target) <= radius;
    if any(mask)
        peak = max(profile(mask));
    else
        peak = NaN;
    end
end
