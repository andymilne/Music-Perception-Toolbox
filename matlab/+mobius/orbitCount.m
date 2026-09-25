function n = orbitCount(r)
%MOBIUS.ORBITCOUNT  |Omega_r|, the orbit table's length, without the table.
%
%   N = MOBIUS.ORBITCOUNT(R) returns the number of orbits in the orbit
%   table at tensor order R (2 <= R <= 12). Cost models need only this
%   count, so it is read from a closed table rather than from the orbit
%   table itself: pricing a route at an order beyond the shipped tables
%   must never trigger a build (hours at r = 9).
%
%   |Omega_r| is the number of orbits of ordered pairs of set partitions
%   of an r-element set under simultaneous relabelling of its elements.
%   An orbit is fixed by the block-intersection matrix of the pair up to
%   row and column permutations, so |Omega_r| counts the non-negative
%   integer matrices with entries summing to r and no zero row or
%   column, up to row and column permutations. The values below come
%   from Burnside's lemma (the Python twin's _orbit_count_burnside) and
%   equal the shipped tables' lengths for r <= 8
%   (tests/test_orbit_count.m).
%
%   Twin of Python mpt._mobius.orbit_count.
%
%   See also MOBIUS.GETORBITTABLE.

    arguments
        r (1,1) {mustBeInteger}
    end
    % Indexed by r + 1, for r = 0..12.
    COUNTS = [1, 1, 4, 10, 33, 91, 298, 910, 3017, 9945, 34207, ...
              119369, 429250];
    if r < 2
        error('mobius:orbitCount:rTooSmall', ...
            'Orbit table requires r >= 2; got r=%d.', r);
    end
    if r > numel(COUNTS) - 1
        error('mobius:orbitCount:rTooLarge', ...
            'Orbit count requested at r=%d, beyond hard cap %d.', ...
            r, numel(COUNTS) - 1);
    end
    n = COUNTS(r + 1);
end
