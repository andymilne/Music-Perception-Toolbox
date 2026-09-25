%% demo_probeTone.m
%  Probe-tone fit to a context: spectral pitch class similarity (SPCS)
%  profiles, with and without event weighting by recency and with
%  harmonic and inharmonic spectra.
%
%  In the probe-tone paradigm a listener hears a context and then a
%  single tone, the probe, and rates how well the probe fits. SPCS models
%  the fit as the cosine similarity of two spectrally enriched, absolute,
%  periodic monad (r = 1) expectation tensors: one of the context, one of
%  the probe. Sweeping the probe across the octave gives a probe-tone
%  profile.
%
%  Sections
%    1. The C-major scale as a context, with Krumhansl and Kessler's
%       (1982) major-key probe-tone ratings (TISMIR article, Figure 4a).
%    2. Porcupine[7] in 22-EDO (EDO: equal division of the octave),
%       probed at the 22 pitch classes of 22-EDO (TISMIR article,
%       Figure 4b).
%    3. A time-ordered context, a melody that moves from C major to G
%       major, carried as a pre-MAET with a pitch and a time attribute.
%       Event weighting by an exponential recency profile over elapsed
%       time models the fading of earlier events from memory; three
%       spectra (harmonic, stretched, and stiff-string) show what
%       inharmonicity does to the profile.
%    4. Continuity: whether each probe continues the melody's final
%       melodic direction.
%
%  Uses: transformAttributes, simMaet (batched-raw, broadcast form, with
%  spectral enrichment via its 'spectrum' option), packPreMaet,
%  weightEvents, unpackPreMaet, continuity, mptDefaults
%  (from the Music Perception Toolbox).
%
%  The Python mirror is demo_probe_tone.py.

prevDefaults = mptDefaults('showHints', false);

%% === User-adjustable parameters ===

% The TISMIR article's settings (Section 2.4): sigma = 10 cents, 16
% harmonics, power-law rolloff rho = 1. SPCS: r = 1, absolute, periodic
% at the octave.
sigma = 10; r = 1; isRel = false; isPer = true; period = 1200;
specHarm = {'harmonic', 16, 'powerlaw', 1};

% Krumhansl and Kessler (1982) major-key probe-tone ratings, C to B.
kkMajor = [6.35 2.23 3.48 2.33 4.38 4.09 2.52 5.19 2.39 3.66 2.29 2.88];
names = {'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'};

% Section 3: two inharmonic spectra (User Guide, Section 7.3.6).
% Stretched: partial n at ratio n^beta, so beta = 1.02 puts the second
% partial at 1224 cents. Stiff string: ratio n*sqrt(1 + B n^2), with
% B = 5e-4 inside the range quoted for piano strings (about 1e-5 to
% 1e-3).
specNames = {'harmonic', 'stretched', 'stiff'};
spectra = {specHarm, ...
           {'stretched', 16, 1.02, 'powerlaw', 1}, ...
           {'stiff', 16, 5e-4, 'powerlaw', 1}};
recencySd = 3;             % recency profile's standard deviation (beats)

% One-cent probe grid for the profile curves. In MATLAB a batch of
% one-element multisets is written as a column padded with NaN to two
% columns, one probe per row (the padding is stripped per row); the
% context, a vector, is broadcast against every row.
fine  = (0:1199).';
chrom = (0:100:1100).';
asRows = @(v) [v, nan(numel(v), 1)];

%% === 1. C major against Krumhansl and Kessler ===

fprintf('=== 1. C-major scale: SPCS against K&K major-key ratings ===\n');
cMajor = [0 200 400 500 700 900 1100];
spcsC = simMaet(cMajor, [], asRows(fine), [], sigma, r, isRel, ...
                isPer, period, 'spectrum', specHarm, 'verbose', false);
spcsC12 = spcsC(1:100:end);
% The squared correlation is the R^2 of the best affine map from the
% ratings to SPCS, the fit reported in the article. (The article's
% figure script turns kernel truncation off; at the default truncation
% the values agree to about 1e-11.)
R = corrcoef(spcsC12(:), kkMajor(:));
r2C = R(1, 2)^2;
ab = polyfit(kkMajor(:), spcsC12(:), 1);
for i = 1:12
    fprintf('  %-3s SPCS = %.3f   K&K = %.2f\n', names{i}, spcsC12(i), ...
            kkMajor(i));
end
fprintf('  R^2 = %.3f   (article: 0.63)\n', r2C);

%% === 2. Porcupine[7] in 22-EDO ===

fprintf('\n=== 2. Porcupine[7] in 22-EDO, 22 probes ===\n');
step = 1200 / 22;
porcupine = [0 4 7 10 13 16 19] * step;        % steps 4333333
spcsP = simMaet(porcupine, [], asRows(fine), [], sigma, r, isRel, ...
                isPer, period, 'spectrum', specHarm, 'verbose', false);
probes22 = (0:21).' * step;
spcsP22 = simMaet(porcupine, [], asRows(probes22), [], sigma, r, ...
                  isRel, isPer, period, 'spectrum', specHarm, ...
                  'verbose', false);
[~, order] = sort(spcsP22, 'descend');
for k = order(1:7).'
    fprintf('  step %2d (%6.1f cents): SPCS = %.4f\n', k - 1, ...
            probes22(k), spcsP22(k));
end
% The seven best-fitting probes are the seven scale degrees, step 13
% (degree 5) first. The article reads degree 1 as a major tonic and
% degree 5 as a minor tonic.

figure('Position', [100 100 1100 400]);
subplot(1, 2, 1);
plot(fine, spcsC, 'LineWidth', 0.8); hold on;
plot(chrom, kkMajor * ab(1) + ab(2), 'o', 'Color', [0.84 0.15 0.16]);
hold off;
title(sprintf('(a) C major, 12-EDO: R^2 = %.2f', r2C));
legend('SPCS', 'K&K (affinely rescaled)', 'Location', 'southeast');
xlabel('probe (cents)'); ylabel('SPCS');
subplot(1, 2, 2);
plot(fine, spcsP, 'LineWidth', 0.8); hold on;
stem(probes22, spcsP22, 'filled'); hold off;
title('(b) Porcupine[7], 22-EDO: the 22 probes');
xlabel('probe (cents)'); ylabel('SPCS');

%% === 3. A time-ordered context: recency and inharmonicity ===

% A melody of 13 events: a C-major phrase, then a phrase in G major
% that ends by rising E-F#-G. Onsets in beats; the final C of the
% first phrase is held for two beats.
midi   = [60 64 67 65 64 62 60 71 69 67 64 66 67];
onsets = [0 1 2 3 4 5 6 8 9 10 11 12 13];
pitch  = transformAttributes(midi, [], {'midi', 'cents'});
pm     = packPreMaet({pitch, onsets});

% Event weighting: a factor from the time attribute (index 2) into the
% pitch attribute's weights (index 1), decaying exponentially before
% the last onset, so that an event's weight falls with its lag behind
% the end of the context. The time attribute is then dropped.
pmRec = weightEvents(pm, 2, 1, onsets(end), 'exponentialBefore', ...
                     'sd', recencySd, 'dropInputAttr', true);
[pRec, wRec] = unpackPreMaet(pmRec);
fprintf('\n=== 3. Melody context: recency weights over elapsed time ===\n');
fprintf('  %s\n', strjoin(compose('%.2f', wRec{1}), ' '));

% Probes: the twelve chromatic pitch classes. Unweighted context
% (every event at weight 1) versus recency-weighted, under each
% spectrum. Each profile is also compared with the K&K major profile
% in C and rotated to G.
ctxNames = {'uniform', 'recency'};
ctxP = {pitch, pRec{1}};
ctxW = {[], wRec{1}};
prof = cell(3, 2);
fprintf('\n  %-10s %-8s  R^2 K&K C  R^2 K&K G  SD\n', 'spectrum', 'weights');
for si = 1:3
    for ci = 1:2
        s = simMaet(ctxP{ci}, ctxW{ci}, asRows(chrom), [], sigma, r, ...
                    isRel, isPer, period, 'spectrum', spectra{si}, ...
                    'verbose', false);
        prof{si, ci} = s;
        Rc = corrcoef(s(:), kkMajor(:));
        Rg = corrcoef(s(:), circshift(kkMajor(:), 7));
        fprintf('  %-10s %-8s  %9.2f  %9.2f  %.3f\n', specNames{si}, ...
                ctxNames{ci}, Rc(1, 2)^2, Rg(1, 2)^2, std(s, 1));
    end
end
for nm = {'F', 'F#'}
    i = find(strcmp(names, nm{1}));
    fprintf('  harmonic, %s: uniform %.3f, recency %.3f\n', nm{1}, ...
            prof{1, 1}(i), prof{1, 2}(i));
end
% Recency moves the harmonic profile from C major towards G major: F
% loses fit and F# gains it. The inharmonic spectra flatten the profile
% (smaller SD across probes) and lower its fit to K&K: their upper
% partials are mistuned from the 12-EDO pitch classes, so partials of
% the context and of in-key probes coincide less, and the spectral
% kinship that separates in-key from out-of-key probes weakens.

%% === 4. Continuity: does the probe continue the melody's direction? ===

% continuity returns, for each query (here each probe, from middle C
% upwards), the expected length of the backward run of same-direction
% intervals that the step from the melody's last note to the probe
% extends, and the run's signed size in cents, under the same 10-cent
% uncertainty. The melody ends by rising (E-F#-G), so only probes above
% G extend it.
[count, mag] = continuity(midi * 100, 6000 + chrom, sigma);
fprintf('\n=== 4. Continuity of each probe after the melody ===\n');
fprintf('  (SPCS: harmonic spectrum, recency-weighted context)\n');
fprintf('  %-5s  %6s  %4s  %6s\n', 'probe', 'SPCS', 'run', 'cents');
for i = 1:12
    fprintf('  %-5s  %6.3f  %4.1f  %6.0f\n', names{i}, prof{1, 2}(i), ...
            count(i), mag(i));
end

figure('Position', [100 100 1100 400]);
subplot(1, 2, 1);
bar(0:11, [prof{1, 1}(:), prof{1, 2}(:)]);
title('(a) Harmonic spectrum: event weighting');
legend(ctxNames);
subplot(1, 2, 2); hold on;
for si = 1:3
    plot(0:11, prof{si, 2}, 'o-');
end
hold off;
title('(b) Recency-weighted: three spectra');
legend(specNames);
for sp = 1:2
    subplot(1, 2, sp);
    set(gca, 'XTick', 0:11, 'XTickLabel', names);
    xlabel('probe'); ylabel('SPCS'); box on;
end

mptDefaults(prevDefaults);
fprintf('\nDone.\n');
