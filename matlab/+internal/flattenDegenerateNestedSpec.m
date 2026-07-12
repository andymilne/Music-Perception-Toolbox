function flat = flattenDegenerateNestedSpec(spec)
%FLATTENDEGENERATENESTEDSPEC  Flat equivalent of a degenerate nested spec.
%
%   FLAT = FLATTENDEGENERATENESTEDSPEC(SPEC) returns the flat spec struct
%   equivalent to a degenerate nested SPEC, or [] when SPEC is not
%   degenerate (or not a nested spec at all).
%
%   A two-level nested spec is *degenerate* when its nesting encodes
%   nothing beyond a tuple of scalars: every inner group is a singleton
%   read whole (inner r = 1; the inner sym flag is vacuous on a
%   singleton), the inner level is absolute (inner rel false), and the
%   outer level reads all groups (outer r = the number of groups). The
%   flat equivalent is struct('r', K, 'sym', sym(outer), 'rel',
%   rel(outer)) over the same K x N value matrix (their densities are
%   identical). bindEvents applied to flat single-slot events produces
%   exactly this form; the isotropic build recognises the same
%   structure downstream (the singleton-group fast path), but the
%   matrix-covariance path must flatten *before* spec normalisation,
%   since whitening and the r == K constraint need the true tuple
%   size. Outer sym/rel are carried through so checkAnisoConstraints
%   can reject them with its canonical messages.
%
%   See also internal.resolveSpecsForKernelCov, bindEvents.

    flat = [];
    if ~isstruct(spec) || ~isfield(spec, 'tags')
        return
    end
    tags = spec.tags;
    if ~isvector(tags)
        tags = tags(:, 1);           % innermost grouping column
    end
    tags = tags(:);
    K = numel(tags);
    rLv = spec.r(:).';
    if ~isfield(spec, 'sym') || ~isfield(spec, 'rel')
        return
    end
    symLv = spec.sym(:).';
    relLv = spec.rel(:).';
    if numel(rLv) ~= 2 || numel(symLv) ~= 2 || numel(relLv) ~= 2
        return
    end
    if numel(unique(tags)) ~= K      % every group a singleton
        return
    end
    if rLv(1) ~= 1 || rLv(2) ~= K    % each read whole, all read
        return
    end
    if logical(relLv(1))             % inner level absolute
        return
    end
    flat = struct('r', double(K), 'sym', logical(symLv(2)), ...
                  'rel', logical(relLv(2)));
    if isfield(spec, 'name') && ~isempty(spec.name)
        flat.name = spec.name;
    end
end
