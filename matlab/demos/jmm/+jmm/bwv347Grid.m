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
%                    BWV 347 has no rests). Built with gridEvents and
%                    spread one column per part.
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
    nParts = numel(categories(t.part));
    grid = gridEvents(t, gridStep);

    % One column per part: the gridded table is long (one row per note
    % per point), and the analyses want it wide.
    N = max(grid.gridIndex);
    times = (0:N - 1) * gridStep;
    pitchesSatb = nan(N, nParts);
    live = ~isnan(grid.noteId);
    pitchesSatb(sub2ind([N, nParts], grid.gridIndex(live), ...
                        double(grid.part(live)))) = grid.pitch(live);
    bars = jmm.bwv347Bar(times);
end
