function [uniqueBlocks, partBlockIdx, mus] = getPartitionBlockStructure(r)
%MOBIUS.GETPARTITIONBLOCKSTRUCTURE  Distinct blocks and reuse map for the set
%   partitions of {1..r}, cached per r.
%
%   Both Möbius point evaluators --- mobius.evalOrbitAbs and the factored
%   strategy of mobius.evalOrbitRel --- form the set-partition sum
%   sum_pi mu(pi) prod_{B in pi} f(B), in which a block B recurs across every
%   partition that contains it, so evaluating each distinct block's factor
%   once and reusing it is a shared acceleration. This accessor supplies the
%   parts that depend only on r (the distinct-block list and the reuse map);
%   the per-block factor f is mode-specific and the combine is shared
%   (mobius.mobiusPartitionCombine).
%
%   Returns:
%     uniqueBlocks  cell of blocks, in first-appearance order.
%     partBlockIdx  cell; partBlockIdx{p} lists the uniqueBlocks indices of
%                   partition p's blocks, in the partition's block order.
%     mus           row vector of Möbius weights, one per partition.

    persistent cache
    % dictionary (R2022b+) — MathWorks-recommended replacement for
    % containers.Map, and materially faster on the hot lookup pattern.
    % This function is called on every entry to mobius.evalOrbitAbs (and
    % the factored branch of mobius.evalOrbitRel), so the per-call lookup
    % cost shows up in every nested Möbius call --- meaningful when the
    % surrounding work is small. An unconfigured dictionary (before any
    % insert) throws on isKey; guard with numEntries. Values are
    % wrapped in a scalar cell to sidestep dictionary's scalar-value
    % restriction (structs with mixed-size fields can fail even when
    % the outer struct itself is scalar).
    if isempty(cache)
        cache = dictionary();
    end
    if numEntries(cache) > 0 && isKey(cache, r)
        stored = cache(r);
        s = stored{1};
        uniqueBlocks = s.uniqueBlocks;
        partBlockIdx = s.partBlockIdx;
        mus = s.mus;
        return;
    end

    partitions = mobius.getSetPartitionsWithMobius(r);
    uniqueBlocks = {};
    partBlockIdx = cell(1, numel(partitions));
    mus = zeros(1, numel(partitions));
    for iPart = 1:numel(partitions)
        blocks = partitions(iPart).blocks;
        bidx = zeros(1, numel(blocks));
        for b = 1:numel(blocks)
            B = blocks{b};
            k = 0;
            for u = 1:numel(uniqueBlocks)
                if isequal(uniqueBlocks{u}, B)
                    k = u;
                    break;
                end
            end
            if k == 0
                uniqueBlocks{end + 1} = B; %#ok<AGROW>
                k = numel(uniqueBlocks);
            end
            bidx(b) = k;
        end
        partBlockIdx{iPart} = bidx;
        mus(iPart) = partitions(iPart).mu;
    end

    s.uniqueBlocks = uniqueBlocks;
    s.partBlockIdx = partBlockIdx;
    s.mus = mus;
    cache(r) = {s};
end
