function [rsCanon, csCanon, MCanon] = canonicalForm(M, rowSums, colSums)
%MOBIUS.CANONICALFORM  Canonicalise a contingency table under
%   row/column permutations within block-size groups.
%
%   [RS_CANON, CS_CANON, M_CANON] = MOBIUS.CANONICALFORM(M, ROWSUMS, COLSUMS)
%   iteratively sorts rows by (size, content) and columns likewise until
%   stable. Two contingency tables related by a permutation that preserves
%   row block sizes and column block sizes share the same canonical form.
%
%   Inputs:
%     M       — qA-by-qB non-negative integer matrix.
%     ROWSUMS — 1-by-qA vector of row block sizes (matches sum(M,2)').
%     COLSUMS — 1-by-qB vector of column block sizes (matches sum(M,1)).
%
%   Outputs:
%     RS_CANON — Canonicalised row block sizes (1-by-qA).
%     CS_CANON — Canonicalised column block sizes (1-by-qB).
%     M_CANON  — Canonicalised matrix.
%
%   The iteration runs for a fixed 20 passes; this is far more than is
%   ever needed for orbit-table sizes (r <= 8) and matches the Python
%   reference implementation exactly.
%
%   See also MOBIUS.ENUMERATECONTINGENCYTABLES.

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

    rsCanon = rs;
    csCanon = cs;
    MCanon = Mw;
end
