%% demo_softeningEquivalences.m
%  Softening an equivalence by pairing attributes.
%
%  Each of the three flags imposes an equivalence exactly: [rel] relative
%  identifies a tuple with its transpositions, [per] periodic identifies a
%  value with its octaves, and [exch] exchangeable identifies a tuple with
%  its reorderings. Softening an equivalence makes it hold in part. The
%  same values are carried twice in one event, on two attributes that
%  agree in every respect but one: the flagged copy (attribute c) carries
%  the flag at width sigma_c, and the unflagged copy (attribute f) omits it
%  at width sigma_f. The unflagged width sets how firmly the equivalence
%  holds: as sigma_f grows, a pair of element multisets the flag
%  identifies (a query and a context) runs from cosine similarity 0 (the
%  identification lost) up to the flagged copy's own value. For every pair
%  compared here the cosine similarity factorizes as
%
%      (flagged copy alone, at sigma_c) * exp(-|d|^2 / (4 sigma_f^2)),
%
%  d being the difference between the two ordered tuples (Online
%  Supplement, Sec. "Softening an equivalence with paired attributes").
%
%  Sections:
%    1. Transposition softened: a motif against its transposition and an
%       altered transposition, [rel] copy paired with an absolute copy;
%       the pairing equals the kernel covariance that kernelCov builds
%       from the supplement's mapping of (sigma_c, sigma_f).
%    2. Octave equivalence softened: [per] copy paired with a non-periodic
%       copy (demo_helixBlend treats this pairing in depth).
%    3. Reordering softened: exchangeable copy paired with an ordered copy.
%    4. Figure: similarity against sigma_f for the three, with the limits.
%
%  Uses: packPreMaet, flatSpecs, simMaet, kernelCov, mptDefaults
%  (from the Music Perception Toolbox).
%
%  The Python mirror is demo_softening_equivalences.py.

clear; close all;

% The dispatcher's per-call announcements are switched off for a tidy
% printout, and restored at the end.
prevDefaults = mptDefaults('showHints', false);
tStart = tic;

SIG_C = 30;                                   % flagged copy's width (cents)
SIG_F = logspace(1, 5, 41);                   % unflagged copy's widths (cents)
SIG_F_TABLE = [100 300 1000 3000 10000];
MOTIF = [6000 6200 6400 6700];                % C D E G (cents)

% Each case: the flagged copy's [rel exch isPer], the query, and two
% contexts that the flag identifies with the query, exactly or nearly.
titles = {'1. Transposition softened ([rel] copy + absolute copy)', ...
          '2. Octave equivalence softened ([per] copy + non-periodic copy)', ...
          '3. Reordering softened (exchangeable copy + ordered copy)'};
flags = {[true false false], [false false true], [false true false]};
queries = {MOTIF, MOTIF(1), MOTIF};
labels = {{'C D E G up a fifth', '... last note 50 cents sharp'}, ...
          {'C one octave up', 'C two octaves up'}, ...
          {'D C E G (neighbours swapped)', 'G D E C (C and G swapped)'}};
contexts = {{MOTIF + 700, MOTIF + [700 700 700 750]}, ...
            {MOTIF(1) + 1200, MOTIF(1) + 2400}, ...
            {MOTIF([2 1 3 4]), MOTIF([4 2 3 1])}};
UNFLAGGED = {false, false, false};

curves = cell(1, 3);
for c = 1:3
    flag = flags{c};
    q = queries{c};
    flaggedCopy = [{SIG_C}, num2cell(flag)];
    fprintf('\n=== %s ===\n\n', titles{c});
    fprintf('  Query [%s] cents; sigma_c = %g cents.\n', ...
            strjoin(arrayfun(@(v) sprintf('%d', v), q, 'UniformOutput', false), ', '), SIG_C);
    fprintf('  %-30s%s%9s%18s\n', 'context', sprintf('%9g', SIG_F_TABLE), ...
            'flagged', '|pair - product|');
    pairs = zeros(2, numel(SIG_F));
    lims = zeros(1, 2);
    for j = 1:2
        x = contexts{c}{j};
        paired = @(s) localSim(q, x, [flaggedCopy; [{s}, UNFLAGGED]]);
        pairs(j, :) = arrayfun(paired, SIG_F);
        tbl = arrayfun(paired, SIG_F_TABLE);
        lims(j) = localSim(q, x, flaggedCopy);           % sigma_f -> infinity
        product = lims(j) * exp(-sum((x - q) .^ 2) ./ (4 * SIG_F .^ 2));
        fprintf('  %-30s%s%9.4f%18.1e\n', labels{c}{j}, sprintf('%9.4f', tbl), ...
                lims(j), max(abs(pairs(j, :) - product)));
    end
    curves{c} = struct('pairs', pairs, 'lims', lims);
    fprintf('  (Columns: sigma_f in cents; ''flagged'' is the flagged copy alone,\n');
    fprintf('  the limit as sigma_f grows. As sigma_f shrinks the pair tends to 0,\n');
    fprintf('  the identification lost; the last column is the largest departure\n');
    fprintf('  from the factorization over the sigma_f sweep.)\n');

    if flag(1)
        % On an ordered, absolute, non-periodic attribute at r = K, the
        % kernel covariance sigma_val^2 I + sigma_shift^2 11' (sigma_int
        % = 0) is the same kernel as the pairing, under the mapping
        % sigma_val = sigma_c sigma_f / sqrt(sigma_c^2 + sigma_f^2),
        % sigma_shift = sigma_f^2 / sqrt(r (sigma_c^2 + sigma_f^2)).
        r = numel(q);
        v = SIG_C ^ 2 + SIG_F .^ 2;
        fprintf('\n  The pair against one absolute attribute with the kernel\n');
        fprintf('  covariance kernelCov(r, sdValue, sdShift) of the mapping\n');
        fprintf('  (the ridge''s condition number grows as sigma_f^2 / sigma_c^2,\n');
        fprintf('  costing a few digits at the widest sigma_f):\n');
        for j = 1:2
            x = contexts{c}{j};
            ridge = zeros(size(SIG_F));
            for i = 1:numel(SIG_F)
                Sigma = kernelCov(r, 'differenced', false, ...
                                  'sdValue', SIG_C * SIG_F(i) / sqrt(v(i)), ...
                                  'sdShift', SIG_F(i) ^ 2 / sqrt(r * v(i)));
                ridge(i) = localSim(q, x, {Sigma, false, false, false});
            end
            fprintf('    %-30s max |pair - kernelCov| = %.1e\n', labels{c}{j}, ...
                    max(abs(pairs(j, :) - ridge)));
        end
    end
end

%% ===== 4. Figure =====

figure('Name', 'Softening an equivalence', 'Position', [100 100 1300 380]);
cols = lines(2);
for c = 1:3
    ax = subplot(1, 3, c);
    h = gobjects(1, 2);
    for j = 1:2
        h(j) = semilogx(ax, SIG_F, curves{c}.pairs(j, :), 'Color', cols(j, :), ...
                        'LineWidth', 1.2);
        hold(ax, 'on');
        yline(ax, curves{c}.lims(j), '--', 'Color', cols(j, :));
    end
    xline(ax, SIG_C, ':', 'Color', [0.5 0.5 0.5]);
    ylim(ax, [-0.05 1.05]);
    title(ax, extractBefore(titles{c}, ' ('), 'FontSize', 10);
    xlabel(ax, '\sigma_f (cents)');
    legend(h, labels{c}, 'Location', 'west', 'FontSize', 8);
    if c == 1
        ylabel(ax, 'cosine similarity to the query');
    end
end
sgtitle(['Softening an equivalence: dashed lines mark the flagged copy ' ...
         'alone (\sigma_f \rightarrow \infty); dotted, \sigma_c'], 'FontSize', 10);

mptDefaults(prevDefaults);
fprintf('\nDone in %.1f s.\n', toc(tStart));


function s = localSim(q, x, copies)
%LOCALSIM  Cosine similarity of query q and context x under the copies.
%   truncationSigmas = Inf sums each kernel out to the toolbox's accuracy
%   floor (1e-12) rather than the default 6 sigmas, so the comparisons
%   also hold for similarities too small for the default to keep.
    s = simMaet(localPreMaet(q, copies), localPreMaet(x, copies), ...
                'truncationSigmas', Inf, 'verbose', false);
end


function pm = localPreMaet(x, copies)
%LOCALPREMAET  One event whose element multiset x is carried by one
%   attribute per copy; each row of the n x 4 cell copies is
%   {sigma, rel, exch, isPer}, the attribute read whole (r = K).
    n = size(copies, 1);
    p = repmat({x(:)}, 1, n);
    per = [copies{:, 4}];
    pm = packPreMaet(p, [], flatSpecs(p, 'r', numel(x), ...
        'rel', [copies{:, 2}], 'exch', [copies{:, 3}], ...
        'sigma', copies(:, 1).', 'isPer', per, 'period', 1200 * per));
end
