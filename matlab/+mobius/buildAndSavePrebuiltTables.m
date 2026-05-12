function buildAndSavePrebuiltTables(maxR)
%MOBIUS.BUILDANDSAVEPREBUILTTABLES  Build and save shipped orbit tables.
%
%   MOBIUS.BUILDANDSAVEPREBUILTTABLES() builds orbit tables for r =
%   2..8 (the default shipped range) and saves them to
%   matlab/+mobius/_orbit_tables/orbit_r{r}.mat. Build times scale
%   with B_r^2; expect r = 8 to take a few minutes.
%
%   MOBIUS.BUILDANDSAVEPREBUILTTABLES(MAXR) builds for r = 2..MAXR.
%
%   Existing files are not overwritten; delete them manually to force a
%   rebuild. This function is intended for release preparation, not
%   runtime invocation; runtime callers should use MOBIUS.GETORBITTABLE.
%
%   See also MOBIUS.BUILDORBITTABLE, MOBIUS.GETORBITTABLE.

    arguments
        maxR (1,1) {mustBeInteger, mustBePositive} = 8
    end

    if maxR < 2
        error('mobius:buildAndSavePrebuiltTables:maxRTooSmall', ...
            'maxR must be at least 2 (orbit machinery is undefined below 2).');
    end

    here = fileparts(mfilename('fullpath'));
    prebuiltDir = fullfile(here, '_orbit_tables');
    if ~isfolder(prebuiltDir)
        mkdir(prebuiltDir);
    end

    for r = 2:maxR
        out = fullfile(prebuiltDir, sprintf('orbit_r%d.mat', r));
        if isfile(out)
            fprintf('  r=%d: already present, skipping.\n', r);
            continue
        end
        fprintf('  r=%d: building... ', r);
        t0 = tic;
        orbit_table = mobius.buildOrbitTable(r); %#ok<NASGU>
        save(out, 'orbit_table', '-v7');
        fprintf('done (%.2fs, %d orbits).\n', toc(t0), numel(orbit_table)); %#ok<NODEF>
    end
end
