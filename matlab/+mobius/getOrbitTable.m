function table = getOrbitTable(r)
%MOBIUS.GETORBITTABLE  Orbit table for tensor order r, with caching.
%
%   TABLE = MOBIUS.GETORBITTABLE(R) returns the orbit table for tensor
%   order R, looking it up in three tiers:
%
%     1. In-memory cache (persistent across calls within the same
%        MATLAB session; cleared by `clear functions`).
%     2. Shipped pre-built tables in matlab/+mobius/_orbit_tables/.
%     3. User disk cache at ~/.mpt/orbit_tables/ (overridable via the
%        MPT_CACHE_DIR environment variable; same parent directory as
%        the Python sibling but distinct files: .mat here, .pkl there;
%        the two do not interoperate).
%     4. Build from scratch via MOBIUS.BUILDORBITTABLE; if R >= 5 the
%        result is also written to the user disk cache.
%
%   R must satisfy 2 <= R <= 12. Below 2 the orbit machinery is
%   undefined (single-block partition); above 12 the build cost is
%   prohibitive (several minutes at r=8, hours beyond).
%
%   See also MOBIUS.BUILDORBITTABLE, MOBIUS.BUILDANDSAVEPREBUILTTABLES.

    arguments
        r (1,1) {mustBeInteger}
    end

    R_HARD_CAP = 12;

    if r < 2
        error('mobius:getOrbitTable:rTooSmall', ...
            'Orbit table requires r >= 2; got r=%d.', r);
    end
    if r > R_HARD_CAP
        error('mobius:getOrbitTable:rTooLarge', ...
            ['Orbit table requested at r=%d, beyond hard cap %d. ', ...
             'Building tables at this order takes prohibitive time and ', ...
             'memory; if you really need this, raise the cap and accept ', ...
             'the cost.'], r, R_HARD_CAP);
    end

    % Tier 1: in-memory persistent cache.
    persistent inMem
    if isempty(inMem)
        inMem = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    end
    key = int32(r);
    if isKey(inMem, key)
        table = inMem(key);
        return
    end

    % Tier 2: shipped pre-built table alongside this module.
    here = fileparts(mfilename('fullpath'));
    prebuilt = fullfile(here, '_orbit_tables', sprintf('orbit_r%d.mat', r));
    if isfile(prebuilt)
        S = load(prebuilt, 'orbit_table');
        table = ensureRecipes(S.orbit_table);
        inMem(key) = table;
        return
    end

    % Tier 3: user disk cache.
    userDir = getenv('MPT_CACHE_DIR');
    if isempty(userDir)
        if ispc
            home = getenv('USERPROFILE');
        else
            home = getenv('HOME');
        end
        if isempty(home)
            home = pwd;  % last-resort fallback
        end
        userDir = fullfile(home, '.mpt', 'orbit_tables');
    end
    userFile = fullfile(userDir, sprintf('orbit_r%d.mat', r));
    if isfile(userFile)
        S = load(userFile, 'orbit_table');
        table = ensureRecipes(S.orbit_table);
        inMem(key) = table;
        return
    end

    % Tier 4: build from scratch (buildOrbitTable embeds recipes).
    maybeWarnBuildCost(r);
    table = mobius.buildOrbitTable(r);
    inMem(key) = table;

    % Best-effort persist to user disk for non-trivial rebuilds.
    if r >= 5
        try
            if ~isfolder(userDir)
                mkdir(userDir);
            end
            orbit_table = table; %#ok<NASGU>
            save(userFile, 'orbit_table', '-v7');
        catch
            % Disk caching is best effort; in-memory cache is still hot.
        end
    end
end


function maybeWarnBuildCost(r)
%MAYBEWARNBUILDCOST  Print a size + time estimate before building an
%   orbit table. Fires only for r beyond the shipped range (r=2..8 in
%   v3; bump SHIPPED_MAX below if more files are added later).
%   Suppressed when the environment variable MPT_NO_BUILD_WARN is set.
%
%   Output goes to stderr (fprintf(2, ...)) so it doesn't contaminate
%   stdout-based pipelines.

    if ~isempty(getenv('MPT_NO_BUILD_WARN'))
        return
    end
    SHIPPED_MAX = 8;   % match Python _ORBIT_R_MAX_SHIPPED
    if r <= SHIPPED_MAX
        return
    end

    % Bell numbers B_r for r = 0..12. The orbit table at order r has
    % roughly B_r^2 / symmetry orbits.
    BELL = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, ...
            678570, 4213597];
    % Rough build-time estimates in seconds, indexed by r. r = 2..8
    % anchored to measured release-prep runs (2024-era laptop, MATLAB
    % R2024a, single-threaded). r >= 9 extrapolated by the empirical
    % ~5x per-r growth observed across the shipped range.
    TIME_S = containers.Map(...
        {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, ...
        {0.21, 0.29, 0.49, 1.04, 4.59, 18.32, 93.64, ...
         500.0, 2500.0, 1.5e4, 1e5});

    if r + 1 <= numel(BELL)
        bellR = BELL(r + 1);    % MATLAB 1-indexing: BELL(r+1) is B_r
    else
        bellR = NaN;
    end
    if isKey(TIME_S, r)
        timeStr = localFormatDuration(TIME_S(r));
    else
        timeStr = '(unknown)';
    end

    fprintf(2, ...
        ['mpt: building orbit table for r=%d (not shipped, not cached).\n', ...
         '     B_r = %s; build cost scales with B_r squared.\n', ...
         '     Estimated build time: ~%s (rough; depends on system).\n', ...
         '     Result will be cached to disk; subsequent calls return ', ...
         'instantly.\n', ...
         '     Suppress this message by setting MPT_NO_BUILD_WARN=1.\n'], ...
        r, ...
        localFormatBig(bellR), ...
        timeStr);
end


function s = localFormatDuration(seconds)
%LOCALFORMATDURATION  Render a build-time estimate in a friendly unit.
    if seconds < 1
        s = sprintf('%.0f ms', seconds * 1000);
    elseif seconds < 60
        s = sprintf('%.0f s', seconds);
    elseif seconds < 3600
        s = sprintf('%.1f min', seconds / 60);
    elseif seconds < 86400
        s = sprintf('%.1f h', seconds / 3600);
    else
        s = sprintf('%.1f days', seconds / 86400);
    end
end


function s = localFormatBig(x)
%LOCALFORMATBIG  Format a big integer with thousands separators.
    if isnan(x)
        s = '?';
    elseif x < 1e4
        s = sprintf('%d', round(x));
    else
        % MATLAB lacks built-in thousands separators; insert commas.
        digits = sprintf('%d', round(x));
        n = length(digits);
        parts = cell(1, ceil(n / 3));
        for i = 1:ceil(n / 3)
            stop = n - (i - 1) * 3;
            start = max(1, stop - 2);
            parts{ceil(n / 3) - i + 1} = digits(start:stop);
        end
        s = strjoin(parts, ',');
    end
end


% =========================================================================
%  Recipe augmentation and versioning: cached tables (pre-built .mat
%  files, user-tier files, in-memory entries) carry recipes from
%  whichever builder produced them. Recipes are (re)built on first
%  load when the fields are missing entirely, or when their embedded
%  version predates mobius.recipeVersion() -- so improvements to the
%  builder's contraction ordering take effect without regenerating the
%  shipped table files. Recipe building is fast (abstract enumeration
%  only, no actual numeric work), so the cost is bounded and paid once
%  per session; subsequent loads use the in-memory cache directly.
% =========================================================================

function table = ensureRecipes(table)
    if isempty(table)
        return
    end
    needsAugment = ~isfield(table, 'recipeIP') ...
                || ~isfield(table, 'recipeGrid') ...
                || ~isfield(table, 'recipeBatched');
    if ~needsAugment
        % Fields are present; keep them only if the first orbit's
        % recipe is non-empty and carries the current builder version.
        first = table(1);
        if isfield(first.recipeIP, 'steps') ...
                && isfield(first.recipeIP, 'version') ...
                && first.recipeIP.version >= mobius.recipeVersion()
            return
        end
    end
    for k = 1:numel(table)
        [rIP, rGrid, rBatched] = mobius.buildOrbitRecipes(table(k));
        table(k).recipeIP = rIP;
        table(k).recipeGrid = rGrid;
        table(k).recipeBatched = rBatched;
    end
end
