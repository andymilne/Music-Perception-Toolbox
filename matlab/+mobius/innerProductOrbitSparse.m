function [val, ratio, termMass] = innerProductOrbitSparse(Ksp, w_A, w_B, r, opts)
%MOBIUS.INNERPRODUCTORBITSPARSE  Sparse-kernel twin of INNERPRODUCTORBIT.
%
%   VAL = MOBIUS.INNERPRODUCTORBITSPARSE(KSP, W_A, W_B, R) evaluates the
%   distinct-index inner product <T_A, T_B> from a spatially-culled
%   sparse kernel KSP (n_A x n_B). The value equals
%   MOBIUS.INNERPRODUCTORBIT on the same (densified) kernel to floating
%   point.
%
%   Each orbit's bipartite graph is contracted by min-degree
%   elimination: a degree-1 node folds into its neighbour as a mat-vec,
%   a degree-2 node becomes a Gram product M1' diag(w) M2 (a sparse
%   matmul). For every shipped tuple size (r = 2..8; graphs no worse than
%   K_{2,m}) this never needs more than a 2-D intermediate. A degree->=3
%   node reverts that orbit to the dense recipe (MOBIUS.EXECUTERECIPE);
%   a per-matrix density guard densifies any Gram that fills in.
%
%   [VAL, RATIO] = MOBIUS.INNERPRODUCTORBITSPARSE(..., ...
%       'returnCancellationRatio', true) additionally returns
%   |sum| / max_orb(|term_orb|), matching INNERPRODUCTORBIT's contract.
%
%   [VAL, RATIO, TERMMASS] = MOBIUS.INNERPRODUCTORBITSPARSE(..., ...
%       'returnTermMass', true) additionally returns the prefactored
%   max_orb(|term_orb|), mirroring MOBIUS.INNERPRODUCTORBITGRID; callers
%   integrating over a u-grid use it to form a mass-aware cancellation
%   diagnostic.
%
%   Name-value options:
%     'prefactor'                (1,1) double, default 1.0.
%     'returnCancellationRatio'  (1,1) logical, default false.
%     'returnTermMass'           (1,1) logical, default false.
%     'densityThresh'            (1,1) double, default 0.34 -- a Gram
%                                denser than this fraction is densified.
%
%   See also MOBIUS.INNERPRODUCTORBIT, MOBIUS.GETORBITTABLE.

    arguments
        Ksp
        w_A (:,1) double
        w_B (:,1) double
        r (1,1) {mustBeInteger}
        opts.prefactor (1,1) double = 1.0
        opts.returnCancellationRatio (1,1) logical = false
        opts.returnTermMass (1,1) logical = false
        opts.densityThresh (1,1) double = 0.34
    end

    table = mobius.getOrbitTable(r);
    total = 0.0;
    maxAbsTerm = 0.0;
    Kd = [];
    for k = 1:numel(table)
        orb = table(k);
        [c, needDense] = localContractSparse(orb, Ksp, w_A, w_B, ...
            opts.densityThresh);
        if needDense
            if isempty(Kd)
                Kd = full(Ksp);
            end
            c = localContractDenseFrom(orb, Kd, w_A, w_B);
        end
        term = orb.weight * orb.mu * c;
        total = total + term;
        at = abs(term);
        if at > maxAbsTerm
            maxAbsTerm = at;
        end
    end

    val = opts.prefactor * total;
    if opts.returnCancellationRatio
        if maxAbsTerm > 0
            ratio = abs(total) / maxAbsTerm;
        else
            ratio = 1.0;
        end
    else
        ratio = [];
    end
    if opts.returnTermMass
        termMass = opts.prefactor * maxAbsTerm;
    else
        termMass = [];
    end
end


function [scalar, needDense] = localContractSparse(orb, K, w_A, w_B, ...
                                                    densityThresh)
%LOCALCONTRACTSPARSE  Contract one orbit graph by min-degree elimination.
%   Returns NEEDDENSE = true (and an unused SCALAR) when a degree->=3
%   node is reached, signalling the caller to recompute densely.

    needDense = false;
    nNodes = orb.qA + orb.qB;
    nodeW = cell(1, nNodes);
    for alpha = 1:orb.qA
        nodeW{alpha} = w_A .^ orb.m_A(alpha);
    end
    for beta = 1:orb.qB
        nodeW{orb.qA + beta} = w_B .^ orb.m_B(beta);
    end
    active = true(1, nNodes);

    % Edges keyed canonically (p < q); matrix oriented rows=p, cols=q.
    eP = zeros(0, 1);
    eQ = zeros(0, 1);
    eM = {};
    nE = size(orb.edges, 1);
    for e = 1:nE
        alpha = orb.edges(e, 1);
        beta  = orb.edges(e, 2);
        m     = orb.edges(e, 3);
        aNode = alpha;                 % A-node id in 1..qA
        bNode = orb.qA + beta;         % B-node id in qA+1..qA+qB (> aNode)
        [eP, eQ, eM] = localAddEdge(eP, eQ, eM, aNode, bNode, ...
            localKpow(K, m));
    end

    scalar = 1.0;
    while any(active)
        act = find(active);
        deg = zeros(1, numel(act));
        for t = 1:numel(act)
            v = act(t);
            deg(t) = sum(eP == v | eQ == v);
        end
        [d, tmin] = min(deg);
        v = act(tmin);

        if d == 0
            scalar = scalar * sum(nodeW{v});
            active(v) = false;
        elseif d == 1
            ei = find(eP == v | eQ == v, 1);
            p = eP(ei); q = eQ(ei); M = eM{ei};
            if p == v
                u = q; vec = M.' * nodeW{v};
            else
                u = p; vec = M * nodeW{v};
            end
            nodeW{u} = nodeW{u} .* vec;
            [eP, eQ, eM] = localRemoveEdge(eP, eQ, eM, ei);
            active(v) = false;
        elseif d == 2
            eis = find(eP == v | eQ == v);
            [M1, u1] = localOrientRows(eP(eis(1)), eQ(eis(1)), eM{eis(1)}, v);
            [M2, u2] = localOrientRows(eP(eis(2)), eQ(eis(2)), eM{eis(2)}, v);
            M2s = localRowScale(nodeW{v}, M2);
            G = M1.' * M2s;                     % (n_u1 x n_u2)
            if issparse(G) && nnz(G) > densityThresh * size(G, 1) * size(G, 2)
                G = full(G);
            end
            [eP, eQ, eM] = localRemoveEdge(eP, eQ, eM, max(eis(1), eis(2)));
            [eP, eQ, eM] = localRemoveEdge(eP, eQ, eM, min(eis(1), eis(2)));
            active(v) = false;
            [eP, eQ, eM] = localAddEdge(eP, eQ, eM, u1, u2, G);
        else
            needDense = true;
            scalar = 0.0;
            return;
        end
    end
end


function c = localContractDenseFrom(orb, Kd, w_A, w_B)
%LOCALCONTRACTDENSEFROM  Dense-recipe fallback for a single orbit.
    nE = size(orb.edges, 1);
    operands = cell(1, orb.qA + orb.qB + nE);
    idx = 1;
    for alpha = 1:orb.qA
        operands{idx} = w_A .^ orb.m_A(alpha); idx = idx + 1;
    end
    for beta = 1:orb.qB
        operands{idx} = w_B .^ orb.m_B(beta); idx = idx + 1;
    end
    for e = 1:nE
        m = orb.edges(e, 3);
        if m == 1
            operands{idx} = Kd;
        else
            operands{idx} = Kd .^ m;
        end
        idx = idx + 1;
    end
    c = mobius.executeRecipe(operands, orb.recipeIP);
end


function Km = localKpow(K, m)
    if m == 1
        Km = K;
    else
        Km = spfun(@(x) x .^ m, K);   % elementwise power on nonzeros
    end
end


function [Mv, u] = localOrientRows(p, q, M, v)
%LOCALORIENTROWS  Return M with node v as its row axis, and the other node.
    if p == v
        Mv = M; u = q;
    else
        Mv = M.'; u = p;
    end
end


function R = localRowScale(wv, M)
%LOCALROWSCALE  diag(wv) * M for sparse or dense M.
    if issparse(M)
        R = spdiags(wv, 0, numel(wv), numel(wv)) * M;
    else
        R = wv .* M;                  % implicit row expansion (wv column)
    end
end


function C = localElemMul(A, B)
%LOCALELEMMUL  A .* B, returning sparse only when both are sparse.
    if issparse(A) && issparse(B)
        C = A .* B;
    else
        if issparse(A), A = full(A); end
        if issparse(B), B = full(B); end
        C = A .* B;
    end
end


function [eP, eQ, eM] = localAddEdge(eP, eQ, eM, a, b, M)
%LOCALADDEDGE  Insert edge (a,b) with M rows=a; merge parallels by product.
    if a <= b
        p = a; q = b; Mc = M;
    else
        p = b; q = a; Mc = M.';
    end
    existing = find(eP == p & eQ == q, 1);
    if isempty(existing)
        eP(end + 1, 1) = p;
        eQ(end + 1, 1) = q;
        eM{end + 1} = Mc;
    else
        eM{existing} = localElemMul(eM{existing}, Mc);
    end
end


function [eP, eQ, eM] = localRemoveEdge(eP, eQ, eM, i)
    eP(i) = [];
    eQ(i) = [];
    eM(i) = [];
end
