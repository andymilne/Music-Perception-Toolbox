function win = winEvents(a, b)
%WINEVENTS  Eighth-note events in [a, b).
%
%   win = jmm.winEvents(a, b)
%
%   Returns a struct with the window's eighth-note events:
%     .times   - 1 x M event times (QN).
%     .chords  - 1 x M cell; entry m is an n x 3 matrix [note id, pitch,
%                sounding fraction] of the notes sounding during the
%                eighth-note event at .times(m): fraction 1 when the note
%                sounds through the eighth, 0.5 when it sounds for a
%                single sixteenth.
%     .w       - 1 x M metric / fermata event weights.
%
%   Twin of bwv_window.win_events in the Python demos.
%
%   See also JMM.BWVWINDOWSTATE, JMM.AGGREGATE.
    S = jmm.bwvWindowState();
    m = (S.e8Times >= a - 1e-9) & (S.e8Times < b - 1e-9);
    win.times = S.e8Times(m);
    win.chords = cell(1, numel(win.times));
    for i = 1:numel(win.times)
        win.chords{i} = localEventNoteFracs(S, win.times(i));
    end
    win.w = S.e8W(m);
end


function out = localEventNoteFracs(S, t)
    % Notes sounding during the eighth-note event at t: [note id, pitch,
    % sounding fraction], one row per note that sounds at all.
    a = S.notes(:, 2); b = S.notes(:, 3);
    frac = zeros(size(a));
    for g = [t, t + S.gridStep]
        frac = frac + 0.5 * ((a - 1e-9 <= g) & (g < b - 1e-9));
    end
    nid = find(frac > 0);
    out = [nid, S.notes(nid, 1), frac(nid)];
end
