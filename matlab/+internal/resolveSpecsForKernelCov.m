function specs = resolveSpecsForKernelCov(specs, sigmaVec)
%RESOLVESPECSFORKERNELCOV  Flatten degenerate nested specs on matrix sigma.
%
%   SPECS = RESOLVESPECSFORKERNELCOV(SPECS, SIGMAVEC) returns the specs
%   cell with every attribute carrying a matrix-valued kernel
%   covariance holding a flat spec: degenerate nested specs (see
%   internal.flattenDegenerateNestedSpec) are replaced by their flat
%   equivalents; a non-degenerate nested spec on a matrix-sigma
%   attribute errors. Attributes with isotropic sigma are left
%   untouched, nested or not. Outer-level sym/rel flags on a flattened
%   spec are rejected downstream by internal.checkAnisoConstraints
%   exactly as on a flat attribute.
%
%   See also internal.flattenDegenerateNestedSpec, buildExpTens.

    if ~iscell(specs) || ~iscell(sigmaVec)
        return
    end
    for a = 1:min(numel(specs), numel(sigmaVec))
        s = specs{a};
        if ~internal.isKernelCov(sigmaVec{a})
            continue
        end
        if ~isstruct(s) || ~isfield(s, 'tags')
            continue
        end
        flat = internal.flattenDegenerateNestedSpec(s);
        if isempty(flat)
            error('mpt:aniso:notDegenerate', ...
                  ['sigma{%d}: a matrix-valued kernel covariance is ' ...
                   'supported on a nested attribute only in the ' ...
                   'degenerate case (two levels, every group a ' ...
                   'singleton read whole, inner level absolute, outer ' ...
                   'level reading all groups), which is order-' ...
                   'isomorphic to a flat tuple and is flattened ' ...
                   'automatically. This spec''s nesting is not ' ...
                   'degenerate.'], a);
        end
        specs{a} = flat;
    end
end
