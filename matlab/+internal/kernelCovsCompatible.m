function tf = kernelCovsCompatible(densX, densY)
%KERNELCOVSCOMPATIBLE  True if two densities carry equal kernel covariances.
%
%   Both absent counts as compatible; one present and one absent, or
%   any per-attribute mismatch of shape or values (to 1e-12 relative),
%   is incompatible.

    hasX = internal.densityHasKernelCov(densX);
    hasY = internal.densityHasKernelCov(densY);
    if ~hasX && ~hasY
        tf = true;
        return;
    end
    if hasX ~= hasY
        tf = false;
        return;
    end
    ax = densX.kernelCov;
    ay = densY.kernelCov;
    if iscell(ax) ~= iscell(ay)
        tf = false;
        return;
    end
    if ~iscell(ax)
        ax = {ax}; ay = {ay};
    end
    if numel(ax) ~= numel(ay)
        tf = false;
        return;
    end
    tf = true;
    for a = 1:numel(ax)
        ea = isempty(ax{a}); eb = isempty(ay{a});
        if ea && eb, continue; end
        if ea ~= eb, tf = false; return; end
        if ~isequal(size(ax{a}), size(ay{a}))
            tf = false; return;
        end
        scale = max(abs(ax{a}(:)));
        if max(abs(ax{a} - ay{a}), [], 'all') > 1e-12 * max(scale, 1)
            tf = false; return;
        end
    end
end
