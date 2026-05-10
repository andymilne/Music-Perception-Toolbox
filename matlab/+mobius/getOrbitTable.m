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


% =========================================================================
%  Backward-compat augmentation: old pre-built tables (shipped before
%  the recipe fields existed) get their recipes computed and attached
%  on first load. Recipe building is fast (abstract enumeration only,
%  no actual numeric work), so the cost is bounded and paid once per
%  session; subsequent loads use the in-memory cache directly. Tables
%  freshly built by mobius.buildOrbitTable already have recipes
%  embedded and pass through unchanged.
% =========================================================================

function table = ensureRecipes(table)
    if isempty(table)
        return
    end
    needsAugment = ~isfield(table, 'recipeIP') ...
                || ~isfield(table, 'recipeGrid') ...
                || ~isfield(table, 'recipeBatched');
    if ~needsAugment
        % Field is present; check the first orbit's recipe is non-empty
        % (so we don't silently re-augment a freshly-built table).
        first = table(1);
        if isfield(first.recipeIP, 'steps')
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
