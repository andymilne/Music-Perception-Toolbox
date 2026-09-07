function agg = aggregate(win)
%AGGREGATE  Merge a window's eighth events to one beat aggregate.
%
%   agg = jmm.aggregate(win)
%
%   One entry per note sounding in the window: a note sounding in both of
%   the beat's events is a single entry whose weights sum (the merging
%   rule is that its pitch persists across the merged events); two
%   simultaneous notes of the same pitch remain two entries (a doubling
%   stays doubled). Each note's weight in an event is the event weight
%   times its sounding fraction; all weights are normalized by the 1.5 a
%   beat carries.
%
%   Input: a window struct from jmm.winEvents. Output: a struct with 1 x n
%   .p (pitches) and .w (weights), in order of first appearance.
%
%   Twin of bwv_window.aggregate in the Python demos.
%
%   See also JMM.WINEVENTS, JMM.BOUNDDENSITY.
    S = jmm.bwvWindowState();
    ids = zeros(1, 0); p = zeros(1, 0); w = zeros(1, 0);
    for m = 1:numel(win.chords)
        ev = win.chords{m};
        for j = 1:size(ev, 1)
            nid = ev(j, 1);
            k = find(ids == nid, 1);
            if isempty(k)
                ids(end + 1) = nid; %#ok<AGROW>
                p(end + 1) = ev(j, 2); %#ok<AGROW>
                w(end + 1) = win.w(m) * ev(j, 3); %#ok<AGROW>
            else
                w(k) = w(k) + win.w(m) * ev(j, 3);
            end
        end
    end
    agg.p = p;
    agg.w = w / S.beatWeightNorm;
end
