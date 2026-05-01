%% demo_windowingReference.m
%  MAET post-tensor windowing: choice of reference point for the offset axis.
%
%  windowedSimilarity returns a windowed-similarity profile as a function of
%  a user-supplied offset delta on each windowed attribute. The offset
%  is measured from a reference point to the window centre. Two
%  reference-point options are provided:
%
%    D (default)  The unweighted column mean of the query's tuple centres
%                 on each attribute. A geometric property of the query's
%                 tuple centres, independent of the tuple weights.
%
%    F (fixed)    A user-supplied constant reference, one vector per
%                 attribute, that does not depend on the query.
%
%  The choice matters most when a pitch attribute has more than one slot
%  per event (chords with exchangeable voices, or partials added by
%  addSpectra). Queries can then differ in:
%
%    - slot count   (e.g., adding partials)
%    - slot values  (e.g., stretching partials)
%    - slot weights (e.g., changing rolloff)
%
%  This demo characterises how the similarity profile responds to each
%  of these kinds of between-query variation, under each reference
%  method, for two baselines: a canonical harmonic query (12 integer-
%  harmonic partials), and a non-harmonic query (12 stretched partials
%  at beta=1.05).
%
%  Findings summary
%  ================
%  Let P* be the absolute pitch at which the similarity profile peaks,
%  and mu_q the query's unweighted pitch centroid. Peak offsets satisfy
%
%      delta*_D = P* - mu_q        (default)
%      delta*_F = P* - ref_fixed   (fixed reference)
%
%  Responses to between-query variation:
%
%    * Slot weights. Neither P* nor mu_q moves. Both methods give
%      identical, stable peak offsets. Holds for any slot structure.
%
%    * Slot values. mu_q moves smoothly with the sweep parameter while
%      P* sits on a branch of the similarity profile that may be pinned
%      locally in absolute pitch. Within a branch, delta*_D drifts;
%      delta*_F stays put. Holds for any slot structure.
%
%    * Slot count. For harmonic queries (slot values at or close to
%      integer-harmonic positions above each fundamental), P* and mu_q
%      co-move closely, so delta*_D is stable. For non-harmonic
%      queries, the two move by different amounts, so delta*_D drifts.
%      delta*_F shifts with P* in both cases.
%
%  The harmonic case is special for slot-count changes because integer-
%  harmonic positions on a log-frequency axis are self-similar under
%  extension.
%
%  Figures
%  =======
%  Figure 1: Profiles at fixed time for four query configurations,
%           under both reference methods.
%  Figure 2: Three sweeps (N, beta, rho) for each baseline as
%           similarity-profile heatmaps (rows: sweep parameter,
%           columns: reference method).
%  Figure 3: Peak absolute pitch P* and query centroid mu_q as a
%           function of the sweep parameter. Peak offsets follow as
%           differences: delta*_D = P* - mu_q, delta*_F = P* - ref_fixed.
%  Figure 4: Stretch sweep under F calibrated to the canonical harmonic
%           query. Partial-alignment branch reads as +700 across the
%           sweep; peak amplitude reflects degree of harmonicity match.
%
%  Uses: buildExpTens, evalExpTens, cosSimExpTens, windowedSimilarity,
%        addSpectra, convertPitch.

%% === User-editable parameters ===
PITCH_SIGMA_CENTS = 12.0;
TIME_SIGMA_SEC    = 0.030;
WIN_SIZE_PITCH_EFF = 400.0;      % cents
WIN_SIZE_TIME     = 25;          % multiples of TIME_SIGMA_SEC (~0.75 s)
WIN_MIX           = 0.5;

N_QUERY_EVENTS = 3;
M4_TIME = 6.5;

HARMONIC_N_PARTIALS = 12;
HARMONIC_ROLLOFF    = 0.5;
INHARMONIC_BETA     = 1.05;

% Sweeps
N_SWEEP    = 12:24;
BETA_HARM  = 0.90:0.01:1.10;
BETA_INH   = 0.95:0.01:1.15;
RHO_SWEEP  = 0.5:0.1:1.5;

OFFSET_GRID = -3000:50:3000;

%% === Context and baseline query ===
events = [0.00 62; 0.50 64; 1.00 65; 1.50 64; 2.00 64; 2.50 65; 3.00 67; ...
          3.25 65; 3.50 64; 4.00 67; 4.50 69; 5.00 70; 6.00 72; 6.25 70; ...
          6.50 69; 7.00 71; 7.50 72; 8.00 74];
ctx_t   = events(:, 1).';
ctx_midi = events(:, 2).';
ctx_p   = convertPitch(ctx_midi, 'midi', 'cents');
q_t     = ctx_t(1:N_QUERY_EVENTS);
q_p     = ctx_p(1:N_QUERY_EVENTS);

dens_c = build_context(ctx_p, ctx_t, HARMONIC_N_PARTIALS, HARMONIC_ROLLOFF, ...
    PITCH_SIGMA_CENTS, TIME_SIGMA_SEC);
spec = struct('size', [WIN_SIZE_PITCH_EFF/PITCH_SIGMA_CENTS, WIN_SIZE_TIME], ...
              'mix',  [WIN_MIX, WIN_MIX]);

dens_q_harm = build_query(q_p, q_t, HARMONIC_N_PARTIALS, HARMONIC_ROLLOFF, 1.0, ...
    PITCH_SIGMA_CENTS, TIME_SIGMA_SEC);
REF_HARM_PITCH = mean(dens_q_harm.Centres{1}, 2);
REF_HARM_TIME  = mean(dens_q_harm.Centres{2}, 2);
REF_HARM = { REF_HARM_PITCH, REF_HARM_TIME };

fprintf('Context: %d events over %.1f s, all %d-partial harmonic.\n', ...
    size(events, 1), ctx_t(end), HARMONIC_N_PARTIALS);
fprintf('Query fundamentals: D-E-F (events 1-%d).\n', N_QUERY_EVENTS);
fprintf('Baseline harmonic query:  12 partials, rho=%.1f.\n', HARMONIC_ROLLOFF);
fprintf('Baseline non-harmonic query: 12 stretched partials, beta=%.2f, rho=%.1f.\n', ...
    INHARMONIC_BETA, HARMONIC_ROLLOFF);
fprintf('Harmonic-calibrated fixed reference: mu_q = %.2f cents.\n\n', ...
    REF_HARM_PITCH);

%% === Figure 1: scenario catalogue ===
fprintf('Figure 1: scenario catalogue ...\n');
scenarios = { ...
    '1. Harmonic, N=12, \rho=0.5',  struct('beta', 1.0,  'n_partials', 12, 'rolloff', 0.5); ...
    '2. Harmonic, N=12, \rho=1.5',  struct('beta', 1.0,  'n_partials', 12, 'rolloff', 1.5); ...
    '3. Stretched, N=12, \rho=0.5', struct('beta', 1.05, 'n_partials', 12, 'rolloff', 0.5); ...
    '4. Harmonic, N=24, \rho=0.5',  struct('beta', 1.0,  'n_partials', 24, 'rolloff', 0.5); ...
};
n_scn = size(scenarios, 1);

fig1 = figure('Name', 'Figure 1: scenarios', 'Position', [100 100 1100 800], ...
              'Color', 'w');
set(fig1, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
y_max = 0;
profs = cell(n_scn, 3);  % {label, mu, p_D, p_F}
for i = 1:n_scn
    kw = scenarios{i, 2};
    dq = build_query(q_p, q_t, kw.n_partials, kw.rolloff, kw.beta, ...
        PITCH_SIGMA_CENTS, TIME_SIGMA_SEC);
    mu = mean(dq.Centres{1});
    p_D = profile_at(dq, dens_c, spec, OFFSET_GRID, M4_TIME, []);
    p_F = profile_at(dq, dens_c, spec, OFFSET_GRID, M4_TIME, REF_HARM);
    profs(i, :) = { mu, p_D, p_F };
    y_max = max([y_max max(p_D) max(p_F)]);
end

for i = 1:n_scn
    mu = profs{i, 1};  p_D = profs{i, 2};  p_F = profs{i, 3};
    ax = subplot(n_scn, 2, 2*(i-1) + 1);
    plot(OFFSET_GRID, p_D, 'b-', 'LineWidth', 1.0); hold on;
    line([0 0], [0 y_max*1.1], 'Color', 'k', 'LineStyle', ':');
    line([700 700], [0 y_max*1.1], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 0.7);
    ylim([0 y_max*1.1]); grid on; box(ax, 'off');
    text(0.98, 0.95, sprintf('ref = %.0f c', mu), 'Units', 'normalized', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
        'FontSize', 8, 'Color', [0.25 0.25 0.25], ...
        'BackgroundColor', 'white');
    if i == 1
        title('Default (ref = query centroid)', 'FontSize', 10, ...
              'FontWeight', 'normal', 'Color', 'k');
    end
    if i == n_scn
        xlabel('offset delta (cents)');
    end
    ylabel(scenarios{i, 1}, 'FontSize', 8, 'FontWeight', 'bold');

    ax = subplot(n_scn, 2, 2*(i-1) + 2);
    plot(OFFSET_GRID, p_F, 'b-', 'LineWidth', 1.0); hold on;
    line([0 0], [0 y_max*1.1], 'Color', 'k', 'LineStyle', ':');
    line([700 700], [0 y_max*1.1], 'Color', 'r', 'LineStyle', '--', 'LineWidth', 0.7);
    ylim([0 y_max*1.1]); grid on; box(ax, 'off');
    text(0.98, 0.95, sprintf('ref = %.0f c', REF_HARM_PITCH), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'top', 'FontSize', 8, 'Color', [0.25 0.25 0.25], ...
        'BackgroundColor', 'white');
    if i == 1
        title('Fixed (ref = harmonic-baseline centroid)', ...
            'FontSize', 10, 'FontWeight', 'normal', 'Color', 'k');
    end
    if i == n_scn
        xlabel('offset delta (cents)');
    end
end
sgtitle('Figure 1. Profiles at t=M4 for four scenarios, under each reference method', ...
        'Color', 'k');

%% === Collect sweeps ===
fprintf('Figure 2/3 sweeps: collecting data ...\n');

harm_label_list = arrayfun(@(n) sprintf('%d', n), N_SWEEP, 'UniformOutput', false);
sweeps = { ...
    'Harm: slot count N',     make_kw_list_N(N_SWEEP, 1.0, HARMONIC_ROLLOFF),                harm_label_list, 'N'; ...
    'Harm: slot values beta', make_kw_list_B(BETA_HARM, HARMONIC_N_PARTIALS, HARMONIC_ROLLOFF), arrayfun(@(b) sprintf('%.2f', b), BETA_HARM, 'UniformOutput', false), 'beta'; ...
    'Harm: slot weights rho', make_kw_list_R(RHO_SWEEP, HARMONIC_N_PARTIALS, 1.0),           arrayfun(@(r) sprintf('%.1f', r), RHO_SWEEP, 'UniformOutput', false), 'rho'; ...
    'Inh: slot count N',      make_kw_list_N(N_SWEEP, INHARMONIC_BETA, HARMONIC_ROLLOFF),   harm_label_list, 'N'; ...
    'Inh: slot values beta',  make_kw_list_B(BETA_INH,  HARMONIC_N_PARTIALS, HARMONIC_ROLLOFF), arrayfun(@(b) sprintf('%.2f', b), BETA_INH, 'UniformOutput', false), 'beta'; ...
    'Inh: slot weights rho',  make_kw_list_R(RHO_SWEEP, HARMONIC_N_PARTIALS, INHARMONIC_BETA), arrayfun(@(r) sprintf('%.1f', r), RHO_SWEEP, 'UniformOutput', false), 'rho'; ...
};

sweep_data = cell(size(sweeps, 1), 1);
abs_grid = 4000:50:13000;
for s = 1:size(sweeps, 1)
    kw_list = sweeps{s, 2};
    nS = numel(kw_list);
    P_D = zeros(nS, numel(OFFSET_GRID));
    P_F = zeros(nS, numel(OFFSET_GRID));
    pk_D = zeros(nS, 1);  pk_F = zeros(nS, 1);
    pk_abs = zeros(nS, 1);  mus = zeros(nS, 1);
    for i = 1:nS
        kw = kw_list{i};
        dq = build_query(q_p, q_t, kw.n_partials, kw.rolloff, kw.beta, ...
            PITCH_SIGMA_CENTS, TIME_SIGMA_SEC);
        mus(i) = mean(dq.Centres{1});
        P_D(i, :) = profile_at(dq, dens_c, spec, OFFSET_GRID, M4_TIME, []);
        P_F(i, :) = profile_at(dq, dens_c, spec, OFFSET_GRID, M4_TIME, REF_HARM);
        p_abs = profile_at(dq, dens_c, spec, abs_grid - mus(i), M4_TIME, []);
        [~, ia] = max(p_abs);  pk_abs(i) = abs_grid(ia);
        [~, iD] = max(P_D(i, :));  pk_D(i) = OFFSET_GRID(iD);
        [~, iF] = max(P_F(i, :));  pk_F(i) = OFFSET_GRID(iF);
    end
    sweep_data{s} = struct('P_D', P_D, 'P_F', P_F, ...
        'pk_D', pk_D, 'pk_F', pk_F, 'pk_abs', pk_abs, 'mus', mus);
    fprintf('  %s: %d steps done\n', sweeps{s, 1}, nS);
end

%% === Figure 2: heatmaps ===
fprintf('Figure 2: sweep heatmaps ...\n');
fig2 = figure('Name', 'Figure 2: heatmaps', 'Position', [100 100 1100 1500], ...
              'Color', 'w');
set(fig2, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
for s = 1:size(sweeps, 1)
    d = sweep_data{s};
    labels = sweeps{s, 3};
    nS = numel(labels);
    y_edges = 1:nS;

    subplot(6, 2, 2*(s-1) + 1);
    imagesc(OFFSET_GRID, y_edges, d.P_D); axis xy;
    hold on;
    plot(d.pk_D, y_edges, 'r.-', 'LineWidth', 0.7, 'MarkerSize', 5);
    line([0 0], [y_edges(1) y_edges(end)], 'Color', 'w', 'LineStyle', ':');
    set(gca, 'YTick', y_edges(1:max(1,floor(nS/12)):end), ...
        'YTickLabel', labels(1:max(1,floor(nS/12)):end), 'FontSize', 7);
    ylabel(sprintf('%s\n%s', sweeps{s, 1}, sweeps{s, 4}), 'FontSize', 9);
    if s == 1
        title('D (default)', 'FontSize', 10, 'FontWeight', 'normal', 'Color', 'k');
    end
    if s == 6
        xlabel('offset delta (cents)');
    end

    subplot(6, 2, 2*(s-1) + 2);
    imagesc(OFFSET_GRID, y_edges, d.P_F); axis xy;
    hold on;
    plot(d.pk_F, y_edges, 'r.-', 'LineWidth', 0.7, 'MarkerSize', 5);
    line([0 0], [y_edges(1) y_edges(end)], 'Color', 'w', 'LineStyle', ':');
    set(gca, 'YTick', y_edges(1:max(1,floor(nS/12)):end), ...
        'YTickLabel', labels(1:max(1,floor(nS/12)):end), 'FontSize', 7);
    if s == 1
        title('F (fixed)', 'FontSize', 10, 'FontWeight', 'normal', 'Color', 'k');
    end
    if s == 6
        xlabel('offset delta (cents)');
    end
end
sgtitle('Figure 2. Similarity-profile heatmaps across sweeps. Red line: peak offset.', ...
        'Color', 'k');

%% === Figure 3: peak_abs and mu_q vs sweep parameter ===
fprintf('Figure 3: peak_abs and mu_q ...\n');
fig3 = figure('Name', 'Figure 3: P* and mu_q', 'Position', [100 100 1100 800], ...
              'Color', 'w');
set(fig3, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
axis_titles = {'slot count N', 'slot values beta', 'slot weights rho'};
for col = 1:2
    if col == 1, base_offset = 0; else, base_offset = 3; end
    for row = 1:3
        ax = subplot(3, 2, 2*(row-1) + col);
        idx = base_offset + row;
        d = sweep_data{idx};
        labels = sweeps{idx, 3};
        xvals = 1:numel(labels);
        plot(xvals, d.mus, 'o-', 'Color', [0.2 0.7 0.2], 'MarkerSize', 4, ...
            'MarkerFaceColor', [0.2 0.7 0.2], 'DisplayName', 'mu_q (centroid)'); hold on;
        plot(xvals, d.pk_abs, 's-', 'Color', [0.8 0.2 0.2], 'MarkerSize', 4, ...
            'MarkerFaceColor', [0.8 0.2 0.2], 'DisplayName', 'P* (peak abs)');
        line([xvals(1) xvals(end)], [REF_HARM_PITCH REF_HARM_PITCH], ...
            'Color', 'b', 'LineStyle', '--', ...
            'DisplayName', sprintf('ref\\_fixed = %.0f', REF_HARM_PITCH));
        set(gca, 'XTick', xvals(1:max(1,floor(numel(xvals)/10)):end), ...
            'XTickLabel', labels(1:max(1,floor(numel(xvals)/10)):end), ...
            'FontSize', 7, 'XTickLabelRotation', 45);
        grid on; box(ax, 'off');
        if row == 1
            if col == 1
                title('Harmonic baseline', 'FontSize', 10, ...
                      'FontWeight', 'normal', 'Color', 'k');
            else
                title('Non-harmonic baseline', 'FontSize', 10, ...
                      'FontWeight', 'normal', 'Color', 'k');
            end
        end
        if col == 1
            ylabel(sprintf('%s\nabsolute pitch (cents)', axis_titles{row}), 'FontSize', 9);
        end
        if row == 1 && col == 1
            legend('Location', 'southeast', 'FontSize', 8);
        end
    end
end
sgtitle(sprintf(['Figure 3. Peak absolute pitch P* and query centroid mu_q ' ...
    'vs sweep parameter.\n' ...
    'Peak offset under default = P* - mu_q; peak offset under fixed = P* - ref\\_fixed.']), ...
    'Color', 'k');

%% === Figure 4: stretched sweep under harmonic-calibrated F ===
fprintf('Figure 4: stretched sweep under harmonic-calibrated F ...\n');
BETA_FIG4 = 0.95:0.01:1.15;
nB = numel(BETA_FIG4);
P_cal = zeros(nB, numel(OFFSET_GRID));
pk_off = zeros(nB, 1);  pk_val = zeros(nB, 1);
for i = 1:nB
    dq = build_query(q_p, q_t, HARMONIC_N_PARTIALS, HARMONIC_ROLLOFF, BETA_FIG4(i), ...
        PITCH_SIGMA_CENTS, TIME_SIGMA_SEC);
    p = profile_at(dq, dens_c, spec, OFFSET_GRID, M4_TIME, REF_HARM);
    P_cal(i, :) = p;
    [pk_val(i), idx] = max(p);
    pk_off(i) = OFFSET_GRID(idx);
end

fig4 = figure('Name', 'Figure 4: F calibrated to harmonic', ...
              'Position', [100 100 1300 500], 'Color', 'w');
set(fig4, 'DefaultAxesXColor', 'k', 'DefaultAxesYColor', 'k');
subplot(1, 2, 1);
imagesc(OFFSET_GRID, BETA_FIG4, P_cal); axis xy;
colorbar;
xlabel('offset delta (cents)'); ylabel('stretch parameter beta');
hold on;
line([700 700], [BETA_FIG4(1) BETA_FIG4(end)], ...
    'Color', 'r', 'LineStyle', '--', 'LineWidth', 1.0);
line([0 0], [BETA_FIG4(1) BETA_FIG4(end)], ...
    'Color', 'w', 'LineStyle', ':', 'LineWidth', 0.5);
title('Profile heatmap under F calibrated to harmonic baseline', ...
    'FontSize', 10, 'FontWeight', 'normal', 'Color', 'k');

ax4b = subplot(1, 2, 2);
yyaxis left;
plot(BETA_FIG4, pk_off, 'o-', 'LineWidth', 1.0, 'MarkerSize', 4); hold on;
line([BETA_FIG4(1) BETA_FIG4(end)], [700 700], ...
    'Color', 'r', 'LineStyle', '--', 'LineWidth', 0.7);
ylabel('peak offset (cents)');
yyaxis right;
plot(BETA_FIG4, pk_val, 's-', 'Color', [0.2 0.7 0.2], 'LineWidth', 1.0, ...
    'MarkerSize', 4, 'MarkerFaceColor', [0.2 0.7 0.2]);
ylabel('peak similarity value', 'Color', [0.2 0.7 0.2]);
xlabel('stretch parameter beta');
title('Peak offset and value vs beta under F (harmonic-calibrated)', ...
    'FontSize', 10, 'FontWeight', 'normal', 'Color', 'k');
grid on; box(ax4b, 'off');
legend({'peak offset (F)', '+700 (P5)', 'peak value (F)'}, ...
    'Location', 'southwest', 'FontSize', 8);

sgtitle(sprintf(['Figure 4. Stretched query swept %.2f-%.2f, ' ...
    'scored under F calibrated to harmonic baseline (ref = %.0f cents)'], ...
    BETA_FIG4(1), BETA_FIG4(end), REF_HARM_PITCH), 'Color', 'k');

fprintf('\nDone.\n');

%% === Helper subfunctions ===
function kw_list = make_kw_list_N(N_vec, beta, rolloff)
    kw_list = cell(numel(N_vec), 1);
    for i = 1:numel(N_vec)
        kw_list{i} = struct('beta', beta, 'n_partials', N_vec(i), 'rolloff', rolloff);
    end
end

function kw_list = make_kw_list_B(beta_vec, N, rolloff)
    kw_list = cell(numel(beta_vec), 1);
    for i = 1:numel(beta_vec)
        kw_list{i} = struct('beta', beta_vec(i), 'n_partials', N, 'rolloff', rolloff);
    end
end

function kw_list = make_kw_list_R(rho_vec, N, beta)
    kw_list = cell(numel(rho_vec), 1);
    for i = 1:numel(rho_vec)
        kw_list{i} = struct('beta', beta, 'n_partials', N, 'rolloff', rho_vec(i));
    end
end

function dens = build_context(p_fund, t_fund, N, rolloff, ps, ts)
    [pf, wf] = addSpectra(p_fund, [], 'harmonic', N, 'powerlaw', rolloff);
    n = numel(p_fund);
    dens = buildExpTens({reshape(pf, n, N).', t_fund}, ...
        {reshape(wf, n, N).', []}, ...
        [ps, ts], [1, 1], [], ...
        [false, false], [false, false], [0, 0], 'verbose', false);
end

function dens = build_query(p_fund, t_fund, N, rolloff, beta, ps, ts)
    if beta == 1.0
        [pf, wf] = addSpectra(p_fund, [], 'harmonic', N, 'powerlaw', rolloff);
    else
        [pf, wf] = addSpectra(p_fund, [], 'stretched', N, beta, 'powerlaw', rolloff);
    end
    n = numel(p_fund);
    dens = buildExpTens({reshape(pf, n, N).', t_fund}, ...
        {reshape(wf, n, N).', []}, ...
        [ps, ts], [1, 1], [], ...
        [false, false], [false, false], [0, 0], 'verbose', false);
end

function prof = profile_at(dens_q, dens_c, spec, off_1d, t_fixed, reference)
    M = numel(off_1d);
    offs = [off_1d(:).'; t_fixed * ones(1, M)];
    if isempty(reference)
        prof = windowedSimilarity(dens_q, dens_c, spec, offs, 'verbose', false);
    else
        prof = windowedSimilarity(dens_q, dens_c, spec, offs, ...
            'reference', reference, 'verbose', false);
    end
end
