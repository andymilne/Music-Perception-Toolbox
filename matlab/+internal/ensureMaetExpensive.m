function dens = ensureMaetExpensive(dens)
%ENSUREMAETEXPENSIVE  Populate per-tuple fields on a skinny density.
%
%   DENS = ENSUREMAETEXPENSIVE(DENS) returns DENS with the per-tuple
%   "expensive" fields populated. If DENS already has those fields
%   (e.g. it came from buildMaet with 'lazy', false), the call is a
%   no-op pass-through.
%
%   This is the helper used by consumer entry points (simMaet,
%   evalMaet, entropyMaet, etc.) on density inputs.
%   buildMaet defaults to 'lazy', true, returning a skinny density
%   that exposes only cheap fields (pAttr, w, sigma, r, isRel,
%   isPer, period, dim, etc.). Orbit-method consumers operate
%   directly on the cheap fields and skip this helper; pairwise/centre
%   consumers prepend a single call to it.
%
%   Cheap fields (always present after buildMaet):
%     tag 'MaetDensity' (single-multiset is the A = N = 1 corner):
%       nAttrs, N, r, K, pAttr,
%       w, sigma, isRel, isPer, period, dim, dimPerAttr
%
%   Expensive fields (populated by this helper):
%       Centres, U_perm, V_comb, wJ, wv_comb, nJ, nK,
%       eventOfJ, eventOfK
%
%   Idempotence: detection is by the presence of `Centres`. Calling on a
%   fully-populated density returns it unchanged.
%
%   See also buildMaet, simMaet, evalMaet.

    if ~isstruct(dens)
        error('ensureMaetExpensive:badInput', ...
              'Input must be a density struct from buildMaet.');
    end
    if ~isfield(dens, 'tag')
        error('ensureMaetExpensive:missingTag', ...
              'Input is not a density struct (no .tag field).');
    end

    % Already populated: return as-is.
    if isfield(dens, 'Centres') && ~isempty(dens.Centres)
        return
    end

    % Forward the stored isExch flag so an ordered ([exch]=0) density does
    % not silently revert to symmetric when its expensive fields are
    % materialised. Older skinny structs without the field default to
    % symmetric (empty -> buildMaet default).
    exchArgs = {};
    if isfield(dens, 'isExch') && ~isempty(dens.isExch)
        exchArgs = {dens.isExch};
    end

    % Forward the per-attribute nesting spec (representation B) so a
    % nested attribute is not silently flattened when its expensive
    % fields are materialised.
    nestedArgs = {};
    if isfield(dens, 'nested') && iscell(dens.nested) ...
            && any(~cellfun(@isempty, dens.nested))
        nestedArgs = {'nested', dens.nested};
    end

    % Names are attribute-indexed and rebuild-invariant; the rebuild does
    % not carry them, so save and restore around it.
    savedNames = {};
    if isfield(dens, 'names')
        savedNames = dens.names;
    end

    % Kernel-covariance metadata (matrix-valued sigma densities store
    % whitened values with sigma = 1; the rebuild is numerically
    % correct on those but does not carry the metadata, so save and
    % restore around it).
    savedCov = []; savedChol = []; hadCov = false;
    if isfield(dens, 'kernelCov')
        savedCov = dens.kernelCov;
        savedChol = dens.kernelChol;
        hadCov = true;
    end

    % Per-attribute wrap. The rebuild below does not carry it, and the
    % default is 'full-image', so without this a density built with
    % 'single-image' silently reverts when its expensive fields are
    % materialised --- changing the measure rather than the speed. It
    % shows up only above the sigma/period threshold, where the two
    % forms diverge, which is why it went unnoticed. Twin of the same
    % carry in internal.prunedMaet.
    savedWrap = {}; hadWrap = false;
    if isfield(dens, 'wrap') && ~isempty(dens.wrap)
        savedWrap = dens.wrap;
        hadWrap = true;
    end

    switch dens.tag
        case 'MaetDensity'
            dens = buildMaet( ...
                dens.pAttr, dens.w, dens.sigma, dens.r, ...
                dens.isRel, dens.isPer, dens.period, exchArgs{:}, ...
                nestedArgs{:}, 'lazy', false, 'verbose', false);
            if ~isempty(savedNames)
                dens.names = savedNames;
            end

        otherwise
            error('ensureMaetExpensive:badTag', ...
                  'Unknown density tag: %s', dens.tag);
    end

    if hadWrap
        dens.wrap = savedWrap;
    end
    if hadCov
        dens.kernelCov = savedCov;
        dens.kernelChol = savedChol;
    end
end
