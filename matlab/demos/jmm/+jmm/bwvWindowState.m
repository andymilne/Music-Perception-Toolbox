function S = bwvWindowState()
%BWVWINDOWSTATE  Shared state of the Analysis 1.3 window helpers.
%
%   S = jmm.bwvWindowState()
%
%   Beat aggregates, nested context and query builders, and pitch-derived
%   flags for Analysis 1.3 (cadence localization in BWV 347) live in the
%   jmm package: jmm.boundContext, jmm.query, jmm.asCompared,
%   jmm.dyadQuery, jmm.sonAt, jmm.isRootPosition,
%   jmm.isSixFour, and jmm.b2bar. This function computes, once, the
%   module-level state they share and returns it as a struct.
%
%   The encoding is the article's (Section "Cadence localization using
%   nested multisets"):
%
%   * The chorale is read as eighth-note events on the half-QN grid, where
%     the coverage weighting is the article's 'fraction of the eighth each
%     note sounds': 1 for a note sounding through the eighth, 1/2 for one
%     sounding a single sixteenth.
%   * Those slices are weighted by metrical position -- the on-beat eighth
%     at 1 and the off-beat at 1/2, normalized to mean one so that a note
%     sounding through the beat still weighs one -- and raised by half
%     under a fermata (jmm.bwv347FermataSpans).
%   * Regridding to the beat combines them: a note sounding in both
%     eighths becomes one row whose weights have been summed, while two
%     simultaneous notes of the same pitch stay two rows, each carrying its
%     own note id, so a doubling stays doubled. A pitch sustained through
%     the beat thus has weight 1 (1.5 under a fermata); an off-beat passing
%     chord enters at a third of that.
%   * A context window pair (or triple, for the three-chord prototypes)
%     is one bound pitch attribute: the inner level is each aggregate's
%     pitch multiset ([exch] = 1, r = rInner), the outer level the L
%     ordered aggregates ([exch] = 0, r = L), taken relative at the outer
%     level alone ([rel] = (0, 1)) and periodic (P = 12, sigma = 0.15). The
%     optional inversion flag is a second, simplex-coded attribute
%     (+/-0.5, sigmaFlag = 0.1) carried by query and context alike.
%
%   Every step is a toolbox call: readScore, gridAttrTable twice (the
%   second regridding the first), preMaetFromAttrTable, selectPreMaet,
%   bindEvents, buildMaet, and simMaet.
%
%   Fields
%     .sigmaPitch (0.15 semitones), .period (12), .sigmaFlag (0.1),
%     .rootYes (+0.5: predicate holds), .rootNo (-0.5: predicate fails)
%                    - the article's kernel parameters.
%     .eighth (0.5)  - the eighth-note event grain, QN.
%     .beatWeightNorm (1.5)
%                    - the weight a beat carries (1 on-beat + 0.5 off-beat).
%     .gridStep      - jmm.gridStepQn().
%     .T0, .T1       - first grid time and the end of the last grid step.
%     .sixteenths    - the chorale on the sixteenth-note grid, for the
%                      sonority lookups of jmm.sonAt.
%     .flags         - the pitch-derived inversion predicates, one value
%                      per beat: .sixFour (read at a three-beat window's
%                      first beat) and .rootPositionNext (the predicate one
%                      beat on, for a two-beat window's resolution).
%     .beatsPm       - the beat aggregates as a pre-MAET: the pitch
%                      attribute of each beat, NaN-padded where beats hold
%                      unequal numbers of notes, with the beat's own time
%                      alongside for locating it.
%     .dyadChords    - the minimal cadential prototype (dyad skeleton):
%                      {[59 65], [60 64]}, B-F -> C-E.
%
%   See also JMM.BOUNDCONTEXT, JMM.QUERY, JMM.DYADQUERY.
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
    notes = jmm.bwv347Notes();
    S.sixteenths = gridAttrTable(notes, S.gridStep);
    S.T0 = min(S.sixteenths.gridOnsetBeats);
    S.T1 = max(S.sixteenths.gridOnsetBeats) + S.gridStep;

    % --- the chorale as beat aggregates -------------------------------------
    eighths = gridAttrTable(notes, S.eighth, 'weights', 'coverage', ...
                            'limits', [S.T0, S.T1]);
    onset = eighths.gridOnsetBeats;
    metric = 0.5 * ones(height(eighths), 1);
    metric(abs(mod(onset, 1.0)) < 1e-9) = 1.0;
    spans = jmm.bwv347FermataSpans();
    fermata = ones(height(eighths), 1);
    for i = 1:height(eighths)
        if any(spans(:, 1) - 1e-9 <= onset(i) & onset(i) < spans(:, 2) - 1e-9)
            fermata(i) = 1.5;
        end
    end
    eighths.weight = eighths.weight .* fermata ...
                     .* metric / (S.beatWeightNorm / 2);
    beatTable = gridAttrTable(eighths, 1.0);

    S.beatsPm = preMaetFromAttrTable(beatTable, 'attributes', { ...
        struct('column', 'pitch', 'sigma', S.sigmaPitch, 'r', 1, ...
               'exch', true, 'isPer', true, 'period', S.period), ...
        struct('column', 'onset', 'sigma', 1.0)}, ...
        'time', 'beats', 'weights', 'weight');

    % --- the pitch-derived inversion predicates, one value per beat ---------
    % Rows rather than columns because the flag is a property of the beat,
    % not of each note in it: a column on the attribute table would arrive
    % with one value per note. Each pairs with the window that reads it -- a
    % three-beat window starts at its own antepenult, so sixFour is read at
    % the window's first beat, while a two-beat window resolves on its
    % second, so rootPositionNext is the predicate one beat on.
    [beatPAttr, ~, beatSpecs] = unpackPreMaet(S.beatsPm);
    beatNames = cellfun(@(sp) sp.name, beatSpecs, 'UniformOutput', false);
    beatTimes = beatPAttr{find(strcmp(beatNames, 'onset'), 1)}(1, :);
    sixFour = zeros(1, numel(beatTimes));
    rootNext = zeros(1, numel(beatTimes));
    % The sonority is read from the local table rather than through
    % jmm.sonAt, which would call back into this function while it is still
    % being built. The last beat has no successor, so its rootPositionNext
    % reads an empty sonority, which is not root position; no window of the
    % sweep starts there.
    for i = 1:numel(beatTimes)
        if jmm.isSixFour(localSonAt(S.sixteenths, beatTimes(i)))
            sixFour(i) = S.rootYes;
        else
            sixFour(i) = S.rootNo;
        end
        if jmm.isRootPosition(localSonAt(S.sixteenths, beatTimes(i) + 1.0))
            rootNext(i) = S.rootYes;
        else
            rootNext(i) = S.rootNo;
        end
    end
    S.flags = struct('sixFour', sixFour, 'rootPositionNext', rootNext);

    % --- the minimal cadential prototype (dyad skeleton) --------------------
    S.dyadChords = {[59 65], [60 64]};     % B-F -> C-E
end


function son = localSonAt(sixteenths, t)
    %localSonAt The sonority sounding at QN time T, from the grid table.
    rows = abs(sixteenths.gridOnsetBeats - t) < 1e-9;
    son = double(sixteenths.pitch(rows)).';
end
