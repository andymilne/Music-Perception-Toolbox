function v = singleMultisetView(dens)
%INTERNAL.SINGLEMULTISETVIEW  Flat single-multiset view of a MaetDensity.
%
%   V = INTERNAL.SINGLEMULTISETVIEW(DENS) presents the A = N = 1 corner
%   of a MaetDensity (see internal.isSingleMultiset) under the flat
%   single-multiset field names --- scalar parameters and bare per-tuple
%   arrays --- so the single-multiset fast kernels read one layout. The
%   density itself carries per-attribute cells and length-1 parameter
%   vectors; this view scalarises the parameters and unwraps the A = 1
%   cell entry, and aliases the perm-side weight/count under the
%   single-multiset names (wJ / w_perm, nJ / nJ_perm).
%
%   The view is a plain struct (MATLAB value semantics), materialised
%   from whatever fields are present: cheap fields always, plus the
%   per-tuple "expensive" fields when the density has been through
%   internal.ensureExpTensExpensive. Callers that need the expensive
%   fields ensure them on DENS first, exactly as on the general MA path.
%
%   Twin of Python mpt._tensor.density._SingleMultisetView /
%   single_multiset_view.

    if ~internal.isSingleMultiset(dens)
        error('mpt:singleMultisetView:notSingleMultiset', ...
            ['internal.singleMultisetView requires a single-multiset ' ...
             'density (one flat attribute, one event); got nAttrs=%d, ' ...
             'N=%d.'], densField(dens, 'nAttrs'), densField(dens, 'N'));
    end

    v = struct();
    v.tag = 'SingleMultisetView';   % self-identifying: a view, not a density

    % --- flat parameters (the A = 1 attribute's one multiset;
    %     scalarise length-1 parameters) ---
    v.p      = dens.pAttr{1}(:);
    v.w      = dens.w{1}(:);
    v.sigma  = dens.sigma(1);
    v.r      = dens.r(1);
    v.isRel  = dens.isRel(1);
    v.isPer  = dens.isPer(1);
    v.period = dens.period(1);
    if isfield(dens, 'isSym') && ~isempty(dens.isSym)
        v.isSym = dens.isSym(1);
    else
        v.isSym = true;
    end
    if isfield(dens, 'dimPerAttr') && ~isempty(dens.dimPerAttr)
        v.dim = dens.dimPerAttr(1);
    else
        v.dim = dens.dim;
    end

    % --- per-attribute wrap opt-in. At the A = 1 corner, unwrap the
    %     single cell entry to a bare char, matching the view's flat-
    %     scalar convention throughout. Consumers that read wrap must
    %     handle both the MA cell layout (dens.wrap{a}) and this flat
    %     char form. ---
    if isfield(dens, 'wrap') && ~isempty(dens.wrap)
        wr = dens.wrap;
        if iscell(wr); wr = wr{1}; end
        v.wrap = char(wr);
    end

    % --- matrix-valued kernel covariance metadata (per-attribute at the
    %     A = 1 corner: unwrap the single entry for the flat consumers) ---
    if isfield(dens, 'kernelCov') && ~isempty(dens.kernelCov)
        kc = dens.kernelCov;
        if iscell(kc); kc = kc{1}; end
        v.kernelCov = kc;
        kch = dens.kernelChol;
        if iscell(kch); kch = kch{1}; end
        v.kernelChol = kch;
    end

    % --- per-tuple "expensive" arrays, present only after
    %     ensureExpTensExpensive. At A = 1 the joint centres/perm/comb
    %     arrays coincide with the single-attribute ones, so unwrap the
    %     single cell entry and alias the perm-side weight/count under the
    %     flat single-multiset names. ---
    if isfield(dens, 'Centres') && ~isempty(dens.Centres)
        v.Centres = unwrap1(dens.Centres);
        v.U_perm  = unwrap1(dens.U_perm);
        v.V_comb  = unwrap1(dens.V_comb);
        v.wJ      = dens.wJ;
        v.w_perm  = dens.wJ;
        v.nJ      = dens.nJ;
        v.nJ_perm = dens.nJ;
        v.wv_comb = dens.wv_comb;
        v.nK      = dens.nK;
    end
end


function out = unwrap1(x)
%UNWRAP1  Return the first cell of a 1-cell array, or x itself if already
%   a bare array. The MA fill stores per-attribute cells; the flat view
%   wants the bare A = 1 entry.
    if iscell(x)
        out = x{1};
    else
        out = x;
    end
end


function val = densField(dens, name)
%DENSFIELD  Safe field read for the error message (dens may be malformed).
    if isstruct(dens) && isfield(dens, name)
        val = dens.(name);
    else
        val = -1;
    end
end
