function [rsOut, csOut, MOut] = greedyForm(M, rowSums, colSums)
%MOBIUS.GREEDYFORM  Cheap pre-bucketing of a contingency table.
%
%   [RS, CS, MG] = MOBIUS.GREEDYFORM(M, ROWSUMS, COLSUMS) iteratively
%   sorts rows by (size, content) and columns likewise until stable.
%
%   This is an orbit INVARIANT but not a canonical form: two tables in
%   the same orbit can settle on different fixed points (first at r = 5),
%   so it is used only to bucket tables before MOBIUS.CANONICALFORM is
%   applied to one representative per bucket. Twin of the Python
%   ``_greedy_form``.
%
%   See also MOBIUS.CANONICALFORM, MOBIUS.BUILDORBITTABLE.

    arguments
        M (:,:) {mustBeInteger, mustBeNonnegative}
        rowSums (1,:) {mustBeInteger, mustBeNonnegative}
        colSums (1,:) {mustBeInteger, mustBeNonnegative}
    end

    rs = rowSums(:)';
    cs = colSums(:)';
    Mw = M;

    % Iterate to fixed point. 20 passes is overkill for r <= 8; matches
    % the Python reference for bit-for-bit parity.
    for iter = 1:20
        % Sort rows by (size, row content). sortrows is stable and
        % respects column order, giving a deterministic tiebreak.
        rowKeys = [rs(:), Mw];
        [~, rowPerm] = sortrows(rowKeys);
        Mw = Mw(rowPerm, :);
        rs = rs(rowPerm);

        % Sort columns by (size, column content).
        colKeys = [cs(:), Mw'];
        [~, colPerm] = sortrows(colKeys);
        Mw = Mw(:, colPerm);
        cs = cs(colPerm);
    end

    rsOut = rs;
    csOut = cs;
    MOut = Mw;
end
