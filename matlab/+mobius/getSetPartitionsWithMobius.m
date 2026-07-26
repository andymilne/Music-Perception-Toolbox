function partitions = getSetPartitionsWithMobius(r)
%MOBIUS.GETSETPARTITIONSWITHMOBIUS  Set partitions of {1..r} with Möbius coeffs.
%
%   PARTITIONS = MOBIUS.GETSETPARTITIONSWITHMOBIUS(R) enumerates set
%   partitions of {1, 2, ..., R} as a struct array with fields
%
%     .blocks  Cell array of integer vectors; blocks{l} lists the
%              slot-indices in block l (sorted ascending).
%     .mu      Möbius coefficient mu(0_hat, pi) for the partition,
%              equal to prod_l (-1)^(m_l - 1) * (m_l - 1)! where m_l
%              is the size of block l.
%
%   The number of partitions equals the Bell number B_R: B_2=2, B_3=5,
%   B_4=15, B_5=52, B_6=203, B_7=877. Results are cached per R across
%   calls within a session; clear the cache with `clear functions`.
%
%   Used by MOBIUS.EVALORBITABS to enumerate the alternating sum that
%   computes the absolute-mode tensor at a query point.

    arguments
        r (1,1) {mustBeInteger, mustBeNonnegative}
    end

    persistent cache
    % Cell array indexed by r. r is a small integer (1..~10 in practice),
    % so direct indexing avoids the containers.Map lookup overhead.
    % Called from every entry to mobius.getPartitionBlockStructure,
    % which in turn is called on every Möbius orbit-abs evaluation, so
    % the lookup path is hot.
    if ~isempty(cache) && numel(cache) >= r && ~isempty(cache{r})
        partitions = cache{r};
        return
    end

    if r == 0
        % Single empty partition with mu = 1.
        partitions = struct('blocks', {{}}, 'mu', 1);
    elseif r == 1
        partitions = struct('blocks', {{1}}, 'mu', 1);
    else
        partitions = enumerateSetPartitions(r);
    end
    % Extend cache if needed and store. r == 0 uses idx 1 (MATLAB is
    % 1-based) --- shift by using max(r, 1) as the index; but since
    % r == 0 is rare and returns a scalar struct, simplest is to
    % branch: skip caching for r == 0 (constant, trivial to
    % re-compute), otherwise cache at index r.
    if r >= 1
        if isempty(cache)
            cache = cell(1, max(r, 4));
        elseif numel(cache) < r
            cache{r} = [];  % triggers auto-extend
        end
        cache{r} = partitions;
    end
end


function out = enumerateSetPartitions(r)
%ENUMERATESETPARTITIONS  Build all B_r partitions via incremental insertion.
%
%   For each new index k = 2..r, insert k into one of the existing
%   blocks or start a new singleton block. This generates each
%   partition exactly once (canonical-form: blocks ordered by the
%   smallest index they contain).

    % Start with the partition having just the first slot.
    states = {{[1]}};
    for k = 2:r
        nStates = numel(states);
        nextStates = cell(0, 1);
        for s = 1:nStates
            blocks = states{s};
            nB = numel(blocks);
            % Add k to each existing block.
            for b = 1:nB
                newBlocks = blocks;
                newBlocks{b} = [newBlocks{b}, k];
                nextStates{end+1} = newBlocks; %#ok<AGROW>
            end
            % Or start a new singleton block.
            newBlocks = blocks;
            newBlocks{end+1} = k; %#ok<AGROW>
            nextStates{end+1} = newBlocks; %#ok<AGROW>
        end
        states = nextStates;
    end

    % Compute mu for each partition.
    nP = numel(states);
    out = repmat(struct('blocks', [], 'mu', 0), nP, 1);
    for s = 1:nP
        blocks = states{s};
        mu = 1;
        for b = 1:numel(blocks)
            m = numel(blocks{b});
            if m > 1
                mu = mu * (-1)^(m - 1) * factorial(m - 1);
            end
        end
        out(s).blocks = blocks;
        out(s).mu = mu;
    end
end
