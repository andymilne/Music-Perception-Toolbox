function V = simplexVertices(N, edgeLength)
%SIMPLEXVERTICES Vertices of a regular (N-1)-simplex centred at the origin.
%
%   V = simplexVertices(N) returns an N-by-(N-1) matrix whose rows are
%   the vertices of a regular (N-1)-simplex in R^{N-1}, centred at the
%   origin, with all pairwise vertex distances (edge lengths) equal to 1.
%
%   V = simplexVertices(N, edgeLength) scales the simplex so that all
%   pairwise vertex distances equal edgeLength. edgeLength must be
%   positive.
%
%   This is the natural numerical encoding of an N-level categorical
%   attribute (voice identity, instrument, articulation, etc.) for the
%   multi-attribute expectation tensor (MAET) framework. Each level is
%   represented by an (N-1)-dimensional coordinate vector --- a row of
%   V --- and the categorical attribute group then carries N-1
%   coordinate sub-attributes sharing a single sigma. Because all
%   vertices are pairwise equidistant, no level is privileged over any
%   other, in contrast to dummy or treatment coding.
%
%   Construction: take the N standard basis vectors of R^N (which lie
%   on the hyperplane sum(x) = 1 and are pairwise equidistant), centre
%   them at the origin, and project onto an orthonormal basis of the
%   (N-1)-dimensional subspace orthogonal to the all-ones vector. The
%   result is independent of the choice of basis up to a rotation,
%   which is irrelevant for downstream MAET computations.
%
%   Inputs:
%     N           - Number of categorical levels. Integer >= 2.
%     edgeLength  - (Optional) Pairwise distance between vertices.
%                   Default is 1. Must be positive.
%
%   Output:
%     V           - N-by-(N-1) matrix; row k is the coordinate vector
%                   for level k.
%
%   Examples:
%     V = simplexVertices(2);   % 2 x 1: collapses to a 1-D pair
%     V = simplexVertices(3);   % 3 x 2: equilateral triangle in R^2
%     V = simplexVertices(4);   % 4 x 3: regular tetrahedron in R^3
%
%     % SATB voice encoding with edge length matched to a chosen sigma:
%     V = simplexVertices(4, 4);
%     % Each row of V is the 3-D coordinate vector for one voice
%     % (S, A, T, or B). Use these as values for the 3 numerical
%     % sub-attributes of the categorical group.
%
%   See also BUILDEXPTENS.

    arguments
        N          (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(N, 2)}
        edgeLength (1,1) double {mustBePositive} = 1
    end

    % Centred standard basis: rows are unit vectors minus the centroid.
    % Pairwise distance between rows is sqrt(2).
    Vc = eye(N) - 1/N;

    % Orthonormal basis of the column space of Vc, which is 1^perp
    % (the (N-1)-dimensional subspace orthogonal to the all-ones vector).
    Q = orth(Vc);

    % Express each row of Vc in this basis. The result has N rows and
    % N-1 columns, with pairwise row distance sqrt(2). Rescale to the
    % requested edge length.
    V = (Vc * Q) * (edgeLength / sqrt(2));
end
