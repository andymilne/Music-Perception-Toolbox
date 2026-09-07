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
%     pitch multiset ([sym] = 1, r = rInner), the outer level the ordered
%     aggregates ([sym] = 0, r = L), taken relative at the outer level
%     alone ([rel] = (0, 1)) and periodic (P = 12, sigma = 0.15). The
%     optional inversion flag is a second, simplex-coded attribute
%     (+/-0.5, sigmaFlag = 0.1) carried by query and context alike.
%
%   All densities are built with the toolbox's bindEvents ->
%   buildExpTens pipeline; similarities use cosSimExpTens. Data come from
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
%     .notes         - M x 3 [pitch, start, end]: the score's notes,
%                      recovered from the sampled grid. One note per
%                      (pitch, contiguous sounding span): within each
%                      part's stream a run of equal pitches across
%                      consecutive grid points is one note (the grid
%                      cannot see a re-articulation, and the merging rule
%                      treats a pitch persisting across consecutive events
%                      as one entry in any case). Simultaneous notes of
%                      the same pitch are distinct notes --- a doubling
%                      stays doubled.
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

    % --- the score's notes, recovered from the sampled grid -----------------
    notes = zeros(0, 3);
    nGrid = size(S.satb, 1);
    for v = 1:size(S.satb, 2)
        stream = S.satb(:, v);
        start = 1;
        for i = 2:(nGrid + 1)
            if i == nGrid + 1 || stream(i) ~= stream(start)
                notes(end + 1, :) = [stream(start), ...
                                     S.T0 + (start - 1) * S.gridStep, ...
                                     S.T0 + (i - 1) * S.gridStep]; %#ok<AGROW>
                start = i;
            end
        end
    end
    S.notes = notes;

    % --- the minimal cadential prototype (dyad skeleton) --------------------
    S.dyadChords = {[59 65], [60 64]};     % B-F -> C-E
end
