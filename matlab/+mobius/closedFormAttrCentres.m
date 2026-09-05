function bundle = closedFormAttrCentres(dens, a)
%CLOSEDFORMATTRCENTRES  Materialised tuple-centres bundle for attribute a.
%
%   Mirror of Python _mobius_inner._closed_form_attr_centres, for flat
%   attributes and for a nested attribute whose co-transposition unit is
%   the whole tuple ('outer') or absent ('absolute'). The nested case is
%   the centres route of INTERNAL.NESTEDCONTRACT's per-attribute
%   dispatch, which reaches this function for a relative attribute
%   whenever the materialised tuple set is cheaper than the
%   translation-grid contraction and the declared measure admits it (the
%   route rule lives in INTERNAL.NESTEDCONTRACT).
%
%   The attribute is rebuilt in isolation as a single-attribute
%   density: the multi-attribute density stores tuple-centres on the
%   joint perm-side index with joint weight products, whereas the
%   per-attribute inner matrix needs the attribute's own tuple
%   enumeration with its own weights. The rebuild is a skinny
%   single-attribute construction (milliseconds), eagerly
%   materialised. A nested attribute is rebuilt through the same
%   entry point with its stored (already normalised) nesting spec
%   forwarded as 'nested', so the tuple enumeration, the perm/comb
%   sides and the centres reduction are literally the build's own.
%
%   Relative attributes store centres in the first-coordinate reduced
%   convention (r_a - 1 rows: U_perm(2:r) - U_perm(1)); absolute
%   attributes store full r_a-tuples. MOBIUS.CLOSEDFORMATTRMATRIXFROM
%   evaluates the matching quadratic form for each convention.
%
%   Ragged (NaN-padded) values are carried by the rebuild, which drops
%   every tuple touching a padded value to zero weight.
%
%   Returns a struct with fields:
%     Centres   (d x nJ)  tuple-centres (d = r_a - 1 rel, r_a abs)
%     wJ        (1 x nJ)  per-tuple weights
%     eventOfJ  (1 x nJ)  event index per tuple
%     isPer, period, r, isRel, sigma, N — attribute parameters.
%     innerBlockSize
%               co-transposition block size s_u for an inner /
%               intermediate [rel] unit, 0 otherwise (the twin of the
%               Python bundle's inner_block_size). Always 0 here: a
%               flat attribute has no block metric, and the nested
%               attributes that reach this function are exactly those
%               INTERNAL.NESTEDCONTRACT accepts, which declines an
%               inner / intermediate unit outright. Carried so a caller
%               can assert it rather than assume it.
%     comb      [] or a struct in the same conventions (Centres, wJ,
%               eventOfJ) holding the comb side, plus mult, the
%               tuple-symmetry orbit size the sum must be scaled by. See
%               LOCALCOMBRESTRICTION below, and the consumer
%               MOBIUS.CLOSEDFORMATTRMATRIXFROM.

    Pa = dens.pAttr{a};
    Wa = dens.w{a};
    sigma_a = dens.sigma(a);
    r_a = dens.r(a);
    isRel_a = logical(dens.isRel(a));
    isPer_a = logical(dens.isPer(a));
    period_a = dens.period(a);

    % The attribute's own [sym] flag rides through the rebuild, as it
    % does in Python (_closed_form_attr_centres passes is_sym_vec[a]).
    % An ordered attribute must not be symmetrised by the rebuild: its
    % perm side is its comb side, and the comb-side restriction below
    % detects that structurally (nJ == nK, not r_a! * nK) and declines.
    % Densities built before isSym existed default to the symmetric
    % reading, unchanged.
    isSym_a = true;
    if isfield(dens, 'isSym') && numel(dens.isSym) >= a
        isSym_a = logical(dens.isSym(a));
    end

    % Nested attribute: forward the stored (already normalised) spec.
    % Its per-level [r] / [sym] / tags carry the geometry, and the
    % attribute-level isRel must be passed false --- exactly as the
    % original build received it --- so that the spec's own [rel]
    % selector derives it and the build's vacuous-level collapse
    % behaves identically to the original build. (Passing true would
    % suppress that collapse and, on a spec the build normalised
    % itself, silently produce a different attribute.)
    spec_a = [];
    if isfield(dens, 'nested') && iscell(dens.nested) ...
            && numel(dens.nested) >= a && ~isempty(dens.nested{a})
        spec_a = dens.nested{a};
    end
    if isempty(spec_a)
        da = buildExpTens({Pa}, {Wa}, sigma_a, r_a, isRel_a, isPer_a, ...
            period_a, isSym_a, 'lazy', false, 'verbose', false);
    else
        da = buildExpTens({Pa}, {Wa}, sigma_a, r_a, false, isPer_a, ...
            period_a, isSym_a, 'nested', {spec_a}, ...
            'lazy', false, 'verbose', false);
    end

    bundle = struct();
    bundle.Centres  = da.Centres{1};
    bundle.wJ       = da.wJ;
    bundle.eventOfJ = da.eventOfJ;
    bundle.isPer    = isPer_a;
    bundle.period   = period_a;
    % Read back from the rebuild rather than from the caller's density:
    % where the build collapsed a vacuous nesting level to a flat
    % attribute, the rebuild's r / isRel are the ones the materialised
    % centres are actually in.
    bundle.r        = double(da.r(1));
    bundle.isRel    = logical(da.isRel(1));
    bundle.sigma    = sigma_a;
    bundle.N        = da.N;
    bundle.innerBlockSize = localInnerBlockSize(da);
    bundle.comb     = localCombRestriction(da, bundle.r, bundle.isRel);
end


function s = localInnerBlockSize(da)
%LOCALINNERBLOCKSIZE  Co-transposition block size s_u, 0 when not
%   block-metric. Twin of the first entry of Python's _inner_r_vec(da).
    s = 0;
    if ~isfield(da, 'nested') || ~iscell(da.nested) || isempty(da.nested)
        return;
    end
    spec = da.nested{1};
    if isempty(spec) || ~isstruct(spec) || ~isfield(spec, 'proj')
        return;
    end
    if any(strcmp(spec.proj, {'inner', 'intermediate'}))
        s = prod(double(spec.r(1:spec.relUnit)));
    end
end


function comb = localCombRestriction(da, r_a, isRel_a)
%LOCALCOMBRESTRICTION  Bulger's restriction for the X side of the
%   centres route. Mirror of Python
%   _mobius_inner._comb_side_restriction (flat attributes; see the note
%   on nested attributes below).
%
%   The tuple kernel depends on the two tuples only through the
%   difference d_i = x_i - y_i, and every quadratic form this route
%   evaluates -- sum_i d_i^2 (absolute, and its per-coordinate
%   wrapped-Gaussian product in the periodic case), sum_i d_i^2 -
%   (sum_i d_i)^2 / r (relative non-periodic) and sum_{i<j}
%   wrap(d_i - d_j)^2 / r (relative periodic) -- is a symmetric function
%   of (d_1, ..., d_r). Permuting *both* tuples by the same permutation
%   permutes d and so leaves the kernel unchanged. The stored anchoring
%   (Centres = U_perm(2:r) - U_perm(1)) costs nothing: each form depends
%   on d only through differences d_i - d_j, which the reduced
%   representation preserves.
%
%   For a nested attribute the same argument runs with S_{r_a} replaced
%   by the iterated wreath product G of INTERNAL.NESTEDORBITMULT: every
%   element of G permutes the leaves within each level-u block and
%   permutes level-u blocks as wholes, so it is a symmetry of the
%   quadratic forms above just as a flat transposition is. (Only the
%   'absolute' and whole-tuple 'outer' units reach here, so the flat
%   forms are the only ones evaluated; see the innerBlockSize note in
%   the header.)
%
%   For a symmetric attribute the build's perm side is the comb side
%   tiled by all |G| group elements -- a free G-orbit per comb
%   column -- and the tuple weights are products over the tuple's atoms,
%   hence constant on each orbit. The Y perm side is that same union of
%   full orbits, so it is G-stable. Fixing an X-side orbit O with
%   representative k_x,
%
%     sum_{j_x in O} sum_{j_y} K(x_{j_x}, y_{j_y})
%       = sum_{g} sum_{j_y} K(g x_{k_x}, y_{j_y})
%       = sum_{g} sum_{j_y} K(x_{k_x}, g^{-1} y_{j_y})
%       = |G| sum_{j_y} K(x_{k_x}, y_{j_y}),
%
%   the middle step by invariance of the kernel and the last by g
%   permuting the Y perm side bijectively. Summing over orbits,
%
%     sum_{j_x in perm} sum_{j_y in perm} K
%       = |G| * sum_{k_x in comb} sum_{j_y in perm} K,
%
%   exactly -- not up to a constant. Restricting the X side to
%   combinations and scaling by |G| costs nK * nJ kernel evaluations in
%   place of nJ * nJ, a factor |G|: 2 at flat r = 2, 24 at flat r = 4,
%   and prod_l r_l!^(nodes_l) over a nested attribute's symmetric levels
%   (8 for r = [2, 2], sym = [1, 1]).
%
%   The level structure is read from DA itself rather than from the
%   caller's spec, because the build is authoritative: it collapses a
%   vacuous nesting level to a flat attribute, and the restriction must
%   follow the tuples that were actually materialised.
%
%   Declined ([], leaving the caller on the unrestricted perm-vs-perm
%   form) when there is no orbit to collapse:
%
%     - r_a < 2 for a flat attribute, or |G| < 2 for a nested one
%       (every level ordered): there is no orbit;
%     - an ordered flat attribute (isSym = false): the build sets the
%       perm side equal to the comb side, so scaling by r_a! would be
%       wrong;
%     - any density whose materialised sides do not satisfy
%       nJ == |G| * nK (with nK > 0), which is exactly the structural
%       condition the identity needs. Checked rather than assumed, so a
%       build that ever departed from the orbit tiling above would
%       decline rather than compute a wrong number. It is also what
%       catches a nested attribute whose ragged groups break the free
%       tiling.

    comb = [];
    if ~internal.combRestrictionEnabled()
        return;
    end
    r_a = double(r_a);
    specDa = [];
    if isfield(da, 'nested') && iscell(da.nested) && ~isempty(da.nested) ...
            && ~isempty(da.nested{1})
        specDa = da.nested{1};
    end
    if isempty(specDa)
        if r_a < 2
            return;
        end
        mult = factorial(r_a);
    else
        mult = internal.nestedOrbitMult(specDa.r, specDa.sym);
        if mult < 2
            return;
        end
    end
    nK = double(da.nK);
    if nK <= 0 || double(da.nJ) ~= mult * nK
        return;      % ordered attribute, or an unexpected tiling
    end
    comb = struct();
    comb.Centres  = localReducedCentresFromValues(da.V_comb{1}, specDa, ...
                                                  r_a, isRel_a);
    comb.wJ       = da.wv_comb;
    comb.eventOfJ = da.eventOfK;
    comb.mult     = mult;
end


function C = localReducedCentresFromValues(V, spec, r_a, isRel_a)
%LOCALREDUCEDCENTRESFROMVALUES  Centres array for a value matrix V
%   (r_a rows), in the same reduction the build applies to the perm
%   side. Mirror of Python
%   _mobius_inner._reduced_centres_from_values, and of the Centres
%   block of BUILDEXPTENS's LOCALFILLMAEXPENSIVE for the attributes
%   that reach the closed form: a whole-tuple relative reading anchors
%   at position 0; absolute keeps the values. Both sides of the overlap
%   array must be in the same coordinates, so the comb side is reduced
%   here exactly as the build reduced the perm side.
%
%   An inner / intermediate co-transposition unit (block-reduced per
%   level) never reaches here: INTERNAL.NESTEDCONTRACT declines an
%   inner unit before any centres bundle is built, the flat MA
%   orchestrator sends only flat attributes, and
%   MOBIUS.CLOSEDFORMATTRMATRIXFROM refuses such a bundle. It is
%   refused here too rather than reduced with a silently wrong metric.

    r_a = double(r_a);
    if ~isempty(spec) && isstruct(spec) && isfield(spec, 'proj') ...
            && any(strcmp(spec.proj, {'inner', 'intermediate'}))
        error('mobius:closedFormAttrCentres:innerUnit', ...
              ['the tuple-centres closed form does not carry an inner ' ...
               '[rel] unit; the nested plan should have declined this ' ...
               'attribute']);
    end
    if isRel_a
        if r_a < 2
            C = zeros(0, size(V, 2));
        else
            C = V(2:r_a, :) - V(1, :);
        end
        return;
    end
    C = V;
end
