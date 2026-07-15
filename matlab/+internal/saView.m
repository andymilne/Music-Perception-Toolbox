function v = saView(dens)
%SAVIEW  Single-collection (SA) field view over an A = N = 1 MaetDensity.
%
%   V = INTERNAL.SAVIEW(DENS) returns a struct exposing the
%   single-attribute field names --- scalar p, w, sigma, r, isRel,
%   isPer, period, isSym, dim, and the per-tuple expensive arrays --- over
%   a MaetDensity at the A = 1, N = 1 corner. Single-collection code
%   (evalExpTens SA path, cosSimExpTens, entropyExpTens, prunedExpTens)
%   reads one layout regardless of whether the density was built by the
%   vector or the multi-attribute path, mirroring the architectural
%   identity that a single-attribute density is a MaetDensity with A = 1.
%
%   The returned struct is a read-only value snapshot; SA consumers do
%   not mutate density fields (verified), so a value view is safe and
%   avoids the overhead of a handle class. Per-tuple fields are surfaced
%   only when present (an inexpensive/lazy density exposes just the
%   cheap parameters).
%
%   Twin of python mpt._tensor.density.sa_view / _SaDensityView.
%
%   See also BUILDEXPTENS, INTERNAL.ENSUREEXPTENSEXPENSIVE.

    if ~isstruct(dens) || ~isfield(dens, 'tag')
        error('mpt:saView:badInput', ...
            'saView requires a tagged density struct.');
    end
    % Idempotent: an ExpTensDensity (legacy skinny SA struct or an
    % existing view) is already the single-collection layout, so return
    % it unchanged. Only a MaetDensity needs wrapping. (Mirrors Python
    % sa_view, which returns ExpTensDensity / existing views unchanged.)
    if strcmp(dens.tag, 'ExpTensDensity')
        v = dens;
        return;
    end
    if ~strcmp(dens.tag, 'MaetDensity')
        error('mpt:saView:notMaet', ...
            'saView requires a MaetDensity or ExpTensDensity; got %s.', ...
            iTagOf(dens));
    end
    if double(dens.nAttrs) ~= 1
        error('mpt:saView:notSingleAttr', ...
            'saView requires a single-attribute (A = 1) density; got A = %d.', ...
            double(dens.nAttrs));
    end

    v = struct();
    v.tag = 'ExpTensDensity';   % SA consumers may still branch on this

    % --- flat (scalar) parameters: element 1 of the per-attribute vectors ---
    v.p      = dens.pAttr{1}(:, 1);
    v.w      = dens.w{1}(:, 1);
    v.sigma  = double(dens.sigma(1));
    v.r      = double(dens.r(1));
    v.isRel  = logical(dens.isRel(1));
    v.isPer  = logical(dens.isPer(1));
    v.period = double(dens.period(1));
    v.isSym  = logical(dens.isSym(1));
    v.dim    = double(dens.dimPerAttr(1));
    v.K      = double(dens.K(1));

    % --- per-tuple expensive arrays (present only when materialised) ---
    % SA and MA densities use the same field names (Centres, wJ, nJ, nK,
    % U_perm, V_comb, wv_comb). Per-attribute fields are 1 x A cells on a
    % multi-attribute density; at A = 1 the SA layout is the single
    % attribute's, so a celled field is unwrapped at index 1 while an
    % already-matrix field passes through unchanged.
    v = iCopyTuple(v, dens, 'Centres', true);
    v = iCopyTuple(v, dens, 'U_perm',  true);
    v = iCopyTuple(v, dens, 'V_comb',  true);
    v = iCopyTuple(v, dens, 'wv_comb', false);
    v = iCopyTuple(v, dens, 'wJ',      false);
    v = iCopyTuple(v, dens, 'nJ',      false);
    v = iCopyTuple(v, dens, 'nK',      false);
    v = iCopyTuple(v, dens, 'eventOfJ', false);
    v = iCopyTuple(v, dens, 'eventOfK', false);

    % SA aliases: the single-attribute cosine path reads w_perm / nJ_perm,
    % which the old SA build set equal to wJ / nJ (the perm-side weight
    % products and count). The multi-attribute build names them wJ / nJ
    % only, so surface the aliases here.
    if isfield(v, 'wJ'); v.w_perm  = v.wJ; end
    if isfield(v, 'nJ'); v.nJ_perm = v.nJ; end

    % --- kernel covariance passthrough (anisotropic densities) ---
    if isfield(dens, 'kernelCov');  v.kernelCov  = dens.kernelCov;  end
    if isfield(dens, 'kernelChol'); v.kernelChol = dens.kernelChol; end
end

function v = iCopyTuple(v, dens, name, perAttr)
%ICOPYTUPLE  Copy an expensive field, unwrapping a 1 x A cell at A = 1.
    if ~isfield(dens, name) || isempty(dens.(name))
        return;
    end
    val = dens.(name);
    if perAttr && iscell(val)
        v.(name) = val{1};
    else
        v.(name) = val;
    end
end

function t = iTagOf(dens)
    if isfield(dens, 'tag')
        t = dens.tag;
    else
        t = '<no tag>';
    end
end
