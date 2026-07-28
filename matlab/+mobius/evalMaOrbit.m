function [total, ratio] = evalMaOrbit(dens, x, opts)
%EVALMAORBIT  Factored Möbius point evaluator for a flat multi-attribute MAET.
%
%   The MAET density of Milne (2026), Eq. (maet-density), is
%
%       f(x_1, ..., x_A) = sum_n  prod_a  [Sym or Ord]^{r_a}(M_{a,n})(x_a),
%
%   an outer sum over events n of the tensor product over attributes, the
%   tensor product acting as the pointwise product of the per-attribute
%   densities, each a function of that attribute's query block x_a alone.
%   Evaluation therefore factorises completely across attributes within an
%   event: there is no joint cross-attribute tuple sum. Each per-attribute
%   factor is an ordinary single-attribute expectation-tensor density,
%   evaluated by the Möbius point evaluators MOBIUS.EVALORBITABS (absolute
%   mode) and MOBIUS.EVALORBITREL (relative mode), which compute the value
%   from the raw values via the set-partition Möbius decomposition ---
%   polynomial in K rather than the O(K^r) of the materialised tuple
%   centres.
%
%   TOTAL = MOBIUS.EVALMAORBIT(DENS, X) evaluates the flat MA density DENS
%   at the query points X (a (D, n_q) matrix, D = sum_a (r_a - isRel_a)),
%   returning a (n_q, 1) column of tensor values. The rows of X are split
%   into per-attribute blocks of width DENS.dimPerAttr(a) in attribute
%   order.
%
%   [TOTAL, RATIO] = MOBIUS.EVALMAORBIT(..., 'returnCancellationRatio', true)
%   also returns the per-query worst-case (minimum across events and
%   attributes) mass-aware cancellation ratio of the underlying Möbius
%   alternating sums.
%
%   Name-value arguments:
%     opts.truncationSigmas   Kernel truncation, resolved against the
%                             accuracy floor (Inf -> the floor width).
%                             Default from mptDefaults. Passed through to
%                             the per-attribute evaluators.
%     opts.kernelPrecision    'double' or 'single'. Default from mptDefaults.
%     opts.returnCancellationRatio  logical (default false).
%
%   Scope: flat (non-nested) attributes. A nested attribute is stitched by
%   contraction elsewhere; its per-attribute density is a recursive
%   construction, not a single Möbius sum.
%
%   Twin of python mpt._tensor._ma_eval_orbit.eval_ma_orbit.
%
%   See also MOBIUS.EVALORBITABS, MOBIUS.EVALORBITREL, EVALEXPTENS.

    arguments
        dens (1,1) struct
        x double
        opts.truncationSigmas (1,1) double = mptDefaults('truncationSigmas')
        opts.kernelPrecision (1,:) char = mptDefaults('kernelPrecision')
        opts.returnCancellationRatio (1,1) logical = false
    end

    A       = double(dens.nAttrs);
    N       = double(dens.N);
    dims    = double(dens.dimPerAttr(:)).';
    sigmaG  = dens.sigma(:).';
    rVec    = dens.r(:).';
    isRelG  = dens.isRel(:).';
    isPerG  = dens.isPer(:).';
    periodG = dens.period(:).';

    if ~ismatrix(x)
        error('mpt:evalMaOrbit:queryShape', ...
            'Query must be 2-D (D, n_q).');
    end
    D = sum(dims);
    if size(x, 1) ~= D
        error('mpt:evalMaOrbit:queryDim', ...
            ['Query has %d rows but the joint effective dimension is ' ...
             'D = %d.'], size(x, 1), D);
    end
    n_q = size(x, 2);

    % Split the query into per-attribute blocks once.
    xBlocks = cell(1, A);
    off = 0;
    for a = 1:A
        xBlocks{a} = x(off + (1:dims(a)), :);
        off = off + dims(a);
    end

    total = zeros(n_q, 1);
    wantRatio = opts.returnCancellationRatio;
    if wantRatio
        ratio = ones(n_q, 1);
    else
        ratio = [];
    end

    kwCommon = {'truncationSigmas', opts.truncationSigmas, ...
                'kernelPrecision', opts.kernelPrecision};

    for n = 1:N
        prodVal = ones(n_q, 1);
        for a = 1:A
            p = double(dens.pAttr{a}(:, n));
            if isfield(dens, 'w') && ~isempty(dens.w)
                w = double(dens.w{a}(:, n));
            else
                w = ones(size(p));
            end
            % Drop zero-padded / NaN values (ragged cardinality).
            live = ~isnan(p);
            p = p(live);
            w = w(live);
            sig = double(sigmaG(a));
            r   = double(rVec(a));
            rel = logical(isRelG(a));
            per = logical(isPerG(a));
            P   = double(periodG(a));
            xa  = xBlocks{a};

            kw = [kwCommon, {'is_per', per, 'period', P}];
            % For abs-per attributes only, honour the density's wrap
            % opt-in. evalOrbitRel does not take wrap (relative-mode
            % is always full-image after v3+).
            if ~rel && isfield(dens, 'wrap') && ~isempty(dens.wrap) ...
                    && a <= numel(dens.wrap)
                kw = [kw, {'wrap', char(dens.wrap{a})}];
            end
            if wantRatio
                if rel
                    [fa, ra] = mobius.evalOrbitRel(p, w, sig, r, xa, ...
                        kw{:}, 'returnCancellationRatio', true);
                else
                    [fa, ra] = mobius.evalOrbitAbs(p, w, sig, r, xa, ...
                        kw{:}, 'returnCancellationRatio', true);
                end
                ratio = min(ratio, ra(:));
            else
                if rel
                    fa = mobius.evalOrbitRel(p, w, sig, r, xa, kw{:});
                else
                    fa = mobius.evalOrbitAbs(p, w, sig, r, xa, kw{:});
                end
            end
            prodVal = prodVal .* fa(:);
        end
        total = total + prodVal;
    end
end
