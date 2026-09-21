function S = bwvWindowState()
%BWVWINDOWSTATE  Shared state of the Analysis 1.4 window helpers.
%
%   S = jmm.bwvWindowState()
%
%   Beat aggregates, nested context and query builders, and pitch-derived
%   flags for Analysis 1.4 (cadence localization in BWV 347) live in the
%   jmm package: jmm.winEvents, jmm.aggregate, jmm.boundDensity,
%   jmm.buildPair, jmm.dyadQuery, jmm.sonAt, jmm.isRootPosition,
%   jmm.isSixFour, and jmm.b2bar. This function computes, once, the
%   module-level state they share (bwv_window.py precomputes it at import)
%   and returns it as a struct.
%
%   The encoding is the article's (Section "Cadence localization using
%   nested multisets"):
%
%   * The chorale is read as eighth-note events on the half-QN grid: each
%     event's chord is the SATB sonority sounding at that eighth, weighted
%     by metrical position (1 on the beat, 0.5 off it) and raised by half
%     again under a fermata (jmm.bwv347FermataSpans).
%   * A window's events are merged to one beat aggregate: within each
%     voice, the weights of equal pitches are summed across the window's
%     events (multiplicities across voices are preserved --- a doubled
%     pitch stays doubled), and the result is normalized by the 1.5 a beat
%     carries. A pitch sustained through the beat thus has weight 1 (1.5
%     under a fermata); an off-beat passing chord enters at half weight.
%   * A context window pair (or triple, for the three-chord prototypes)
%     is one bound pitch attribute: the inner level is each aggregate's
%     pitch multiset ([exch] = 1, r = rInner), the outer level the ordered
%     aggregates ([exch] = 0, r = L), taken relative at the outer level
%     alone ([rel] = (0, 1)) and periodic (P = 12, sigma = 0.15). The
%     optional inversion flag is a second, simplex-coded attribute
%     (+/-0.5, sigmaFlag = 0.1) carried by query and context alike.
%
%   All densities are built with the toolbox's bindEvents ->
%   buildMaet pipeline; similarities use simMaet. Data come from
%   jmm.bwv347Grid (the bundled MusicXML read with readScore).
%
%   Fields
%     .sigmaPitch (0.15 semitones), .period (12), .sigmaFlag (0.1),
%     .rootYes (+0.5: predicate holds), .rootNo (-0.5: predicate fails)
%                    - the article's kernel parameters.
%     .eighth (0.5)  - the eighth-note event grain, QN.
%     .beatWeightNorm (1.5)
%                    - the weight a beat carries (1 on-beat + 0.5 off-beat).
%     .gridStep      - jmm.gridStepQn().
%     .times, .satb, .bars
%                    - jmm.bwv347Grid (played-through, repeats expanded).
%     .T0, .T1       - first grid time and the end of the last grid step.
%     .fermataSpans  - jmm.bwv347FermataSpans.
%     .e8Times, .e8W - the eighth-note event times and their metric /
%                      fermata weights. An event's sounding pitches are
%                      all pitches sounding during the eighth --- a voice
%                      moving at the sixteenth level contributes both of
%                      its pitches, each at the event's weight.
%     .e8Events      - 1 x M cell, one entry per eighth-note event: an
%                      n x 3 matrix [note id, pitch, sounding fraction]
%                      of the notes sounding during that eighth, from
%                      gridEvents at the eighth grain under the coverage
%                      weighting. Simultaneous notes of the same pitch
%                      stay distinct --- a doubling stays doubled ---
%                      because each row carries its own note id.
%     .dyadChords    - the minimal cadential prototype (dyad skeleton):
%                      {[59 65], [60 64]}, B-F -> C-E.
%
%   Twin of the module-level state of bwv_window.py in the Python demos.
%
%   See also JMM.WINEVENTS, JMM.AGGREGATE, JMM.BOUNDDENSITY, JMM.DYADQUERY.
    persistent cached
    if isempty(cached)
        cached = localBuild();
    end
    S = cached;
end


function S = localBuild()
    % --- kernel parameters (the article's) ----------------------------------
    S.sigmaPitch = 0.15;      % semitones
    S.period     = 12.0;      % octave (MIDI semitones)
    S.sigmaFlag  = 0.1;       % simplex-coded two-level flag
    S.rootYes    = +0.5;      % flag level: predicate holds
    S.rootNo     = -0.5;      % flag level: predicate fails

    S.eighth = 0.5;           % the eighth-note event grain, QN
    S.beatWeightNorm = 1.5;   % the weight a beat carries (1 on-beat + 0.5 off-beat)
    S.gridStep = jmm.gridStepQn();

    % --- chorale (played-through, repeats expanded) -------------------------
    [S.times, S.satb, S.bars] = jmm.bwv347Grid();
    S.T0 = S.times(1);
    S.T1 = S.times(end) + S.gridStep;      % end of the last grid step
    S.fermataSpans = jmm.bwv347FermataSpans();

    % Eighth-note events: time and metric/fermata weight.
    S.e8Times = S.T0:S.eighth:(S.T1 - S.eighth + 1e-9);
    onBeat = abs(mod(S.e8Times, 1.0)) < 1e-8;
    S.e8W = ones(size(S.e8Times)) * 0.5;
    S.e8W(onBeat) = 1.0;
    under = false(size(S.e8Times));
    for i = 1:numel(S.e8Times)
        t = S.e8Times(i);
        under(i) = any(S.fermataSpans(:, 1) - 1e-9 <= t & ...
                       t < S.fermataSpans(:, 2) - 1e-9);
    end
    S.e8W(under) = S.e8W(under) * 1.5;

    % --- the notes sounding in each eighth, with their sounding fractions ---
    % gridEvents on the eighth grain does this directly: the coverage
    % weighting is the article's 'fraction of the eighth each note sounds',
    % 1 for a note sounding through the eighth and 0.5 for one sounding a
    % single sixteenth.
    gridE8 = gridEvents(jmm.bwv347Notes(), S.eighth, ...
                        'weights', 'coverage', 'limits', [S.T0, S.T1]);
    live = ~isnan(gridE8.noteId);
    gIdx = gridE8.gridIndex(live);
    gNid = gridE8.noteId(live);
    gPit = gridE8.pitch(live);
    gFrac = gridE8.weight(live);
    S.e8Events = cell(1, numel(S.e8Times));
    for i = 1:numel(S.e8Times)
        m = gIdx == i;
        S.e8Events{i} = [gNid(m), gPit(m), gFrac(m)];
    end

    % --- the minimal cadential prototype (dyad skeleton) --------------------
    S.dyadChords = {[59 65], [60 64]};     % B-F -> C-E
end
