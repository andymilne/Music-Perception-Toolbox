function [times, pitchesSatb, bars] = bwv347Grid(gridStep)
%BWV347GRID  BWV 347 on the sixteenth-note grid.
%
%   [times, pitchesSatb, bars] = jmm.bwv347Grid()
%   [times, pitchesSatb, bars] = jmm.bwv347Grid(gridStep)
%
%   The chorale, played through with the bars 1--4 repeat expanded, is
%   sampled on the sixteenth-note grid: the pitch sounding in each of the
%   four voices at every grid point.
%
%   Outputs
%     times        - 1 x N grid times in quarter notes, 0 to ~68 QN (272
%                    points at the default step of jmm.gridStepQn()).
%     pitchesSatb  - N x 4 MIDI pitch sounding in soprano, alto, tenor,
%                    bass at each grid point (NaN where a voice rests;
%                    BWV 347 has no rests).
%     bars         - 1 x N played-through bar number: 0 for the
%                    one-quarter pickup, then 1--17.
%
%   Twin of jmm_data.bwv347_grid in the Python demos.
%
%   See also JMM.BWV347NOTES, JMM.BWV347BAR.
    if nargin < 1 || isempty(gridStep)
        gridStep = jmm.gridStepQn();
    end
    t = jmm.bwv347Notes();
    nParts = numel(t.partNames);
    tEnd = max(t.onsetBeats + t.durationBeats);
    % The half-open range [0, tEnd) stepped at gridStep (numpy arange).
    times = (0:(ceil(tEnd / gridStep) - 1)) * gridStep;
    N = numel(times);
    pitchesSatb = nan(N, nParts);
    for part = 1:nParts
        m = t.part == part;
        on = t.onsetBeats(m);
        off = on + t.durationBeats(m);
        pit = t.pitch(m);
        for i = 1:N
            g = times(i);
            k = find((on <= g + 1e-9) & (g < off - 1e-9), 1);
            if ~isempty(k)
                pitchesSatb(i, part) = pit(k);
            end
        end
    end
    bars = jmm.bwv347Bar(times);
end
