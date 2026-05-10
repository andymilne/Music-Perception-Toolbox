function n = labelledPairsRealisingM(M)
%MOBIUS.LABELLEDPAIRSREALISINGM  Number of labelled partition-pairs with
%   contingency table exactly M.
%
%   N = MOBIUS.LABELLEDPAIRSREALISINGM(M) returns the number of labelled
%   set-partition pairs (pi_A, pi_B) on r elements (where r = sum(M(:)))
%   whose contingency table equals M, for the fixed row and column
%   ordering of M.
%
%   For a fixed ordering this is r! / prod(M_ij!).
%
%   Example:
%     mobius.labelledPairsRealisingM([2 0; 0 2]) == 6   % = 4! / (2! 2!)
%
%   See also MOBIUS.ENUMERATECONTINGENCYTABLES, MOBIUS.CANONICALFORM.

    arguments
        M (:,:) {mustBeInteger, mustBeNonnegative}
    end

    total = sum(M(:));
    denom = prod(factorial(M(:)));
    n = factorial(total) / denom;
end
