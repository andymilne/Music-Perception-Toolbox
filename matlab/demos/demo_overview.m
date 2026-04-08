%% demo_overview.m
%  Quick tour of the Music Perception Toolbox.
%  Sections follow the same order as the User Guide.
%
%  Uses the son clave rhythm [0, 3, 6, 10, 12] in a 16-step cycle and
%  the diatonic scale [0, 2, 4, 5, 7, 9, 11] in 12-EDO as running
%  examples.
%
%  Uses: convertPitch, addSpectra, cosSimExpTens, templateHarmonicity,
%        tensorHarmonicity, spectralEntropy, roughness, balanceCircular,
%        evennessCircular, coherence, sameness, nTupleEntropy, meanOffset,
%        edges, markovS
%  (from the Music Perception Toolbox).

%% === 1. Pitch / frequency conversion (User Guide §6.6) ===

fprintf('=== Pitch conversion ===\n');
fprintf('  MIDI 60 = %.2f Hz\n', convertPitch(60, 'midi', 'hz'));
fprintf('  440 Hz  = %.0f cents\n', convertPitch(440, 'hz', 'cents'));
fprintf('  440 Hz  = %.2f ERB-rate\n', convertPitch(440, 'hz', 'erb'));

%% === 2. Spectral enrichment (User Guide §6.2) ===

fprintf('\n=== Spectral enrichment ===\n');
chord = [0; 400; 700];
[p, w] = addSpectra(chord, [], 'harmonic', 8, 'powerlaw', 1);
fprintf('  3 pitches × 8 harmonics = %d partials\n', numel(p));
fprintf('  First 8 offsets: %s\n', mat2str(round(p(1:8)', 1)));

%% === 3. Expectation tensors and SPCS (User Guide §6.1, §3.3) ===

fprintf('\n=== Spectral pitch class similarity ===\n');
scale_cents = [0, 200, 400, 500, 700, 900, 1100];
chords = {[0, 400, 700], [0, 300, 700], [0, 300, 600]};
chordNames = {'Major', 'Minor', 'Dim'};

% Add harmonic spectra (24 partials, 1/n rolloff)
[scale_p, scale_w] = addSpectra(scale_cents, [], 'harmonic', 24, 'powerlaw', 1);

for i = 1:numel(chords)
    [chord_p, chord_w] = addSpectra(chords{i}, [], 'harmonic', 24, 'powerlaw', 1);
    s = cosSimExpTens(scale_p, scale_w, chord_p, chord_w, ...
                      10, 1, false, true, 1200, 'verbose', false);
    fprintf('  Diatonic vs %-5s triad: %.3f\n', chordNames{i}, s);
end

%% === 4. Consonance and harmonicity (User Guide §6.3) ===

fprintf('\n=== Harmonicity and entropy (JI major triad) ===\n');
ji_triad = [0, 386.31, 701.96];
spec = {'harmonic', 24, 'powerlaw', 1};

[hMax, hEnt] = templateHarmonicity(ji_triad, [], 12, ...
    'chordSpectrum', spec);
fprintf('  Template harmonicity (hMax):     %.4f\n', hMax);
fprintf('  Template harmonicity (hEntropy): %.4f\n', hEnt);

h = tensorHarmonicity(ji_triad, [], 12, 'spectrum', spec);
fprintf('  Tensor harmonicity:              %.4f\n', h);

H = spectralEntropy(ji_triad, [], 12, 'spectrum', spec);
fprintf('  Spectral entropy:                %.4f\n', H);

fprintf('\n=== JI vs 12-EDO comparison ===\n');
edo_triad = [0, 400, 700];
for name = {"JI", "12-EDO"}
    if strcmp(name{1}, 'JI')
        triad = ji_triad;
    else
        triad = edo_triad;
    end
    [hMax, ~] = templateHarmonicity(triad, [], 12, 'chordSpectrum', spec);
    H = spectralEntropy(triad, [], 12, 'spectrum', spec);
    fprintf('  %-6s  hMax=%.4f  specEntropy=%.4f\n', name{1}, hMax, H);
end

fprintf('\n=== Roughness ===\n');
p_cents = convertPitch([60, 64, 67], 'midi', 'cents');
[p_r, w_r] = addSpectra(p_cents, [], 'harmonic', 8, 'powerlaw', 1);
f_hz = convertPitch(p_r, 'cents', 'hz');
r = roughness(f_hz, w_r);
fprintf('  C major triad (8 harmonics): roughness = %.4f\n', r);

%% === 5. Balance and evenness (User Guide §6.4) ===

fprintf('\n=== Balance and evenness ===\n');

diat = [0, 2, 4, 5, 7, 9, 11];
fprintf('  Diatonic scale [0,2,4,5,7,9,11] in 12-EDO:\n');
fprintf('    Balance:  %.3f\n', balanceCircular(diat, [], 12));
fprintf('    Evenness: %.3f\n', evennessCircular(diat, 12));

clave = [0, 3, 6, 10, 12];
fprintf('  Son clave [0,3,6,10,12] in 16:\n');
fprintf('    Balance:  %.3f\n', balanceCircular(clave, [], 16));
fprintf('    Evenness: %.3f\n', evennessCircular(clave, 16));

%% === 6. Scale and rhythm structure (User Guide §6.5) ===

fprintf('\n=== Scale structure (diatonic) ===\n');
[c, nc] = coherence(diat, 12);
[sq, nd] = sameness(diat, 12);
fprintf('  Coherence: %.3f (%d failure)\n', c, nc);
fprintf('  Sameness:  %.3f (%d ambiguity)\n', sq, nd);

H1 = nTupleEntropy(diat, 12, 1);
H2 = nTupleEntropy(diat, 12, 2);
fprintf('  1-tuple entropy: %.3f\n', H1);
fprintf('  2-tuple entropy: %.3f\n', H2);

fprintf('\n=== Rhythm structure (son clave) ===\n');
[c, nc] = coherence(clave, 16);
[sq, nd] = sameness(clave, 16);
fprintf('  Coherence: %.3f (%d failures)\n', c, nc);
fprintf('  Sameness:  %.3f (%d ambiguities)\n', sq, nd);

H1 = nTupleEntropy(clave, 16, 1);
H2 = nTupleEntropy(clave, 16, 2);
fprintf('  1-tuple entropy: %.3f\n', H1);
fprintf('  2-tuple entropy: %.3f\n', H2);

h = meanOffset(clave, [], 16);
fprintf('  Mean offset: %s\n', mat2str(round(h, 3)));

edg = edges(clave, [], 16);
fprintf('  Edges: %s\n', mat2str(round(edg, 3)));

y = markovS(clave, [], 16);
fprintf('  Markov(3): %s\n', mat2str(round(y, 3)));

fprintf('\nDone.\n');
