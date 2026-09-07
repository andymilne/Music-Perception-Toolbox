function pp = pianoPhase()
%PIANOPHASE  Symbolic rendering of Reich's Piano Phase for Analyses 3.1--3.3.
%
%   pp = jmm.pianoPhase()
%
%   Symbolic rendering of Reich's Piano Phase (1967), to the manuscript's
%   specification (Section on the Piano Phase encoding):
%
%   * Both pianos play the twelve-note cell E5, F#5, B5, C#6, D6, F#5, E5,
%     C#6, B5, F#5, D6, C#6 in even notes. Piano 1 holds a fixed
%     inter-onset interval throughout; Piano 2's twelve shifts advance the
%     inter-voice phase k (in note-durations) from one integer to the next
%     across each shift and hold it between shifts.
%   * Each advance interpolates k with a smoothstep (the cubic
%     s(x) = 3x^2 - 2x^3 for x in [0, 1]), an S-shaped ramp beginning and
%     ending with zero slope. Piano 2's instantaneous inter-onset
%     interval, set by the phase's rate of change, dips below the base
%     interval while k is moving and returns to it on each hold.
%   * The base inter-onset interval is 137.85 ms, the piece ~617 s
%     (~8,965 events over the two voices), and the peak tempo deviation
%     1.75% --- which fixes the shift duration: the smoothstep's peak
%     slope is 1.5/T, so T = 1.5 * baseIoi / 0.0175 ~= 11.8 s.
%
%   The manuscript's schedule of hold and accelerando lengths is
%   transcribed from a reference recording (Steve Reich Ensemble, Early
%   Works, Nonesuch 1987); that transcription is not reproduced here, so
%   this reconstruction approximates it with a uniform schedule (leadS,
%   gapS below) matching the published figures' shift centres. All other
%   quantities follow the manuscript exactly.
%
%   The rendering is computed once (persistent) and returned as a struct:
%
%     .cell          - 1 x 12 canonical cell, MIDI semitones.
%     .nc            - pulses per cell (12).
%     .baseIoi       - steady inter-onset interval, seconds (0.13785).
%     .peakDev       - peak tempo deviation (0.0175).
%     .nShifts       - twelve shifts, one whole cell of phase.
%     .nRepsV1       - Piano-1 cells in the rendered piece (373).
%     .shiftDur      - accelerando duration, seconds (~11.8).
%     .leadS, .gapS  - the uniform schedule (unison lead, hold between shifts).
%     .cellDur       - one cell, seconds; .tEnd - the piece, seconds.
%     .shiftStarts   - 1 x 12 accelerando starts, seconds.
%     .shiftCentres  - 1 x 12 accelerando centres, seconds
%                      (piano_phase.shift_centre_times).
%     .lagAt         - function handle: phase k (in pulses) at time x
%                      measured in Piano-1 CELLS (piano_phase.lag_at).
%     .voice1, .voice2
%                    - structs with 1 x M .pitch (MIDI) and .onset
%                      (seconds) (piano_phase.render_voice).
%     .piece         - both voices pooled and time-sorted: .pitch, .onset,
%                      .voice (1 or 2) (piano_phase.render_piece).
%
%   Twin of piano_phase.py in the Python demos.
    persistent cached
    if isempty(cached)
        cached = localRender();
    end
    pp = cached;
end


function pp = localRender()
    % --- canonical cell (E5, F#5, B5, C#6, D6, F#5, E5, C#6, B5, F#5, D6, C#6) ---
    pp.cell = [76 78 83 85 86 78 76 85 83 78 86 85];
    pp.nc = 12;

    % --- manuscript constants ---------------------------------------------
    pp.baseIoi = 0.13785;      % seconds; base inter-onset interval (manuscript)
    pp.peakDev = 0.0175;       % peak tempo deviation (manuscript: 1.75%)
    pp.nShifts = 12;           % one whole cell of phase across the piece
    pp.nRepsV1 = 373;          % Piano-1 cells: 373 * 12 * 137.85 ms ~= 617 s

    % Smoothstep peak slope is 1.5/T, and the peak tempo deviation is
    % baseIoi * (dk/dt)_max, so the shift duration follows from the manuscript:
    pp.shiftDur = 1.5 * pp.baseIoi / pp.peakDev;    % ~= 11.8 s per accelerando

    % Uniform stand-in for the recording-transcribed schedule (see header):
    pp.leadS = 40.0;           % seconds of unison before the first shift
    pp.gapS = 35.8;            % seconds of steady phase between shifts

    pp.cellDur = pp.nc * pp.baseIoi;
    pp.tEnd = pp.nRepsV1 * pp.cellDur;

    pp.shiftStarts = pp.leadS + (0:(pp.nShifts - 1)) * (pp.shiftDur + pp.gapS);
    pp.shiftCentres = pp.shiftStarts + pp.shiftDur / 2.0;

    % Phase k of Piano 2 ahead of Piano 1 (pulses) at time x_cells, measured
    % in Piano-1 cells: the sum of the twelve smoothstep ramps.
    shiftStarts = pp.shiftStarts; shiftDur = pp.shiftDur; cellDur = pp.cellDur;
    pp.lagAt = @(xCells) localLagAt(xCells, cellDur, shiftStarts, shiftDur);

    % --- Piano 2's cumulative pulse count: Phi2(t) = t/baseIoi + k(t) ------
    gridDt = 0.0002;                                  % 0.2 ms inversion grid
    % numpy arange(0, tEnd + gridDt, gridDt): ceil((tEnd + gridDt) / gridDt)
    % points at multiples of the step.
    tg = (0:(ceil((pp.tEnd + gridDt) / gridDt) - 1)) * gridDt;
    phi2 = tg / pp.baseIoi + pp.lagAt(tg / pp.cellDur);

    % --- voice 1: fixed inter-onset interval ------------------------------
    m1 = 0:(pp.nRepsV1 * pp.nc - 1);
    pp.voice1.pitch = pp.cell(mod(m1, pp.nc) + 1);
    pp.voice1.onset = m1 * pp.baseIoi;

    % --- voice 2: invert the (increasing) phase on the grid ---------------
    nPulses = floor(phi2(end));
    m2 = 0:(nPulses - 1);
    pp.voice2.pitch = pp.cell(mod(m2, pp.nc) + 1);
    pp.voice2.onset = interp1(phi2, tg, m2, 'linear');

    % --- both voices pooled and time-sorted --------------------------------
    pitch = [pp.voice1.pitch, pp.voice2.pitch];
    onset = [pp.voice1.onset, pp.voice2.onset];
    voice = [ones(size(pp.voice1.onset)), 2 * ones(size(pp.voice2.onset))];
    [~, order] = sort(onset);                         % stable in MATLAB
    pp.piece.pitch = pitch(order);
    pp.piece.onset = onset(order);
    pp.piece.voice = voice(order);
end


function k = localLagAt(xCells, cellDur, shiftStarts, shiftDur)
    t = xCells * cellDur;
    k = zeros(size(t));
    for s = shiftStarts
        x = min(max((t - s) / shiftDur, 0.0), 1.0);
        k = k + (3.0 * x.^2 - 2.0 * x.^3);
    end
end
