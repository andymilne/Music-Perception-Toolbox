function table = buildOrbitTable(r)
%MOBIUS.BUILDORBITTABLE  Orbit table for tensor order r by direct enumeration.
%
%   TABLE = MOBIUS.BUILDORBITTABLE(R) returns the orbit table for the
%   Mobius-Bulger orbit decomposition at tensor order R, as a struct
%   array with one element per orbit class. Each element has fields:
%
%     .weight  Number of (pi_A, pi_B) labelled partition pairs in the
%              orbit. Orbit weights sum to B_R^2.
%     .mu      Mobius coefficient mu(0_hat, pi_A) * mu(0_hat, pi_B);
%              shared across the orbit (depends only on block sizes).
%     .m_A     Block-size profile of the A-side canonical
%              representative; 1-by-qA row vector.
%     .m_B     Block-size profile of the B-side canonical
%              representative; 1-by-qB row vector.
%     .edges   Non-zero entries of the canonical multiplicity matrix as
%              an nEdges-by-3 matrix; each row is [alpha, beta, m] where
%              alpha and beta are 1-based block indices and m is the
%              entry value.
%     .qA      numel(m_A); cached for downstream convenience.
%     .qB      numel(m_B); cached for downstream convenience.
%
%   Algorithm:
%     1. Enumerate integer partitions of R for m_A and m_B.
%     2. For each (m_A, m_B), enumerate contingency tables with those
%        margins, and group by canonical form.
%     3. The orbit weight for each canonical class is
%        (tables_in_orbit * labelled_pairs(M)) / (aut(m_A) * aut(m_B)).
%
%   Orbit weights sum to B_R^2 by construction; this is checked in
%   matlab/tests/test_mobius_orbit_table.m.
%
%   For r in 2..6 use MOBIUS.GETORBITTABLE instead, which loads the
%   shipped pre-built tables and caches in memory across calls.
%
%   See also MOBIUS.GETORBITTABLE, MOBIUS.BUILDANDSAVEPREBUILTTABLES.

    arguments
        r (1,1) {mustBeInteger, mustBePositive}
    end

    % Initialise as a typed 0-by-0 struct array so concatenation below
    % is consistent. Recipe fields are populated below; the recipe
    % builder runs once per orbit at table-build time and the results
    % are embedded so MOBIUS.INNERPRODUCTORBIT* can call
    % MOBIUS.EXECUTERECIPE at runtime without re-deriving the
    % contraction graph (which is the dominant per-call cost of a
    % dynamic contraction; see the REFERENCE.CONTRACT test oracle). The three recipes correspond to the three
    % consumer call patterns (IP / Grid / PwBatched).
    template = struct( ...
        'weight', 0, ...
        'mu', 0, ...
        'm_A', zeros(1, 0), ...
        'm_B', zeros(1, 0), ...
        'edges', zeros(0, 3), ...
        'qA', 0, ...
        'qB', 0, ...
        'recipeIP', struct(), ...
        'recipeGrid', struct(), ...
        'recipeBatched', struct());
    table = repmat(template, 0, 1);

    partitions = mobius.integerPartitions(r);

    for ia = 1:numel(partitions)
        m_A = partitions{ia};
        muA = mobius.mobiusForBlocksizes(m_A);
        autA = mobius.autSize(m_A);
        qA = numel(m_A);

        for ib = 1:numel(partitions)
            m_B = partitions{ib};
            muB = mobius.mobiusForBlocksizes(m_B);
            autB = mobius.autSize(m_B);
            qB = numel(m_B);

            % Enumerate contingency tables and canonicalise each.
            T = mobius.enumerateContingencyTables(m_A, m_B);
            nT = numel(T);
            if nT == 0
                continue
            end
            % Flat key: [rs_canon (1xqA) | cs_canon (1xqB) | M_canon(:)' (1xqA*qB)]
            keyDim = qA + qB + qA * qB;
            flatKeys = zeros(nT, keyDim);
            for k = 1:nT
                [rsC, csC, McC] = mobius.canonicalForm(T{k}, m_A, m_B);
                flatKeys(k, :) = [rsC, csC, McC(:)'];
            end
            % Group by canonical form. We use the default sorted-unique
            % (no 'stable') because (i) orbit order within a bucket
            % does not affect any downstream computation, which
            % iterates all orbits; (ii) sorted order is well-defined
            % across MATLAB releases; (iii) 'stable' is observed to
            % return an empty index vector on duplicate rows under
            % Octave, breaking development-time testing there.
            [uniqKeys, ~, ic] = unique(flatKeys, 'rows');
            counts = accumarray(ic, 1);

            for j = 1:size(uniqKeys, 1)
                rsCanon = uniqKeys(j, 1:qA);
                csCanon = uniqKeys(j, qA + 1:qA + qB);
                Mcanon = reshape(uniqKeys(j, qA + qB + 1:end), qA, qB);

                lp_M = mobius.labelledPairsRealisingM(Mcanon);
                weight = counts(j) * lp_M / (autA * autB);

                % Edges: non-zero entries of M as [alpha, beta, m] rows.
                % find() returns row vectors when its input is a row
                % vector and column vectors otherwise; force columns so
                % the concatenation is always nEdges-by-3.
                [alphas, betas] = find(Mcanon > 0);
                alphas = alphas(:);
                betas = betas(:);
                lin = sub2ind(size(Mcanon), alphas, betas);
                ms = Mcanon(lin);
                ms = ms(:);
                edges = [alphas, betas, ms];

                entry = struct( ...
                    'weight', weight, ...
                    'mu', muA * muB, ...
                    'm_A', rsCanon, ...
                    'm_B', csCanon, ...
                    'edges', edges, ...
                    'qA', qA, ...
                    'qB', qB, ...
                    'recipeIP', struct(), ...
                    'recipeGrid', struct(), ...
                    'recipeBatched', struct());
                [entry.recipeIP, entry.recipeGrid, entry.recipeBatched] = ...
                    mobius.buildOrbitRecipes(entry);
                table(end + 1, 1) = entry; %#ok<AGROW>
            end
        end
    end
end
