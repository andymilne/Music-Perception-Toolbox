function [terms, bulger, info] = nestedCostTerms(densX, densY, a, ...
        truncationSigmas, skipXX, skipYY)
%INTERNAL.NESTEDCOSTTERMS  Analytic cost terms for one attribute.
%
%   [TERMS, BULGER, INFO] = INTERNAL.NESTEDCOSTTERMS(DENSX, DENSY, A)
%   returns the analytic quantities the nested-attribute cost model is
%   fitted against, for attribute A of the pair (DENSX, DENSY):
%
%     TERMS   struct with fields centres, taugrid, contract_relnonper and
%             contract -- each summed over the inner matrices a cosine
%             computes (xy always; xx and yy unless their skip flags are
%             set), each matrix carrying its own event-pair count.
%     BULGER  the joint tuple-pair kernel size of the WHOLE density (all
%             attributes), which is what the joint-tuple enumeration
%             builds for the same three matrices. Always the full three
%             matrices, whatever SKIPXX / SKIPYY say: the enumeration
%             memoises under its own cache key, so it takes its own skip
%             flags, which the caller applies to the per-matrix
%             components INFO.bulgerXY / bulgerXX / bulgerYY.
%     INFO    the counts the terms were built from: mPermX, mCombX,
%             mPermY, mCombY, restrictedX, restrictedY, workX, workY,
%             nTau, nLine, totalOrder, Nx, Ny, bulgerXY, bulgerXX,
%             bulgerYY.
%
%   TRUNCATIONSIGMAS is optional ([] = the mptDefaults default); it
%   enters through the quadrature node counts. SKIPXX / SKIPYY are
%   optional (both false by default, which is what the calibration
%   harness wants: it times cold densities and so computes all three
%   matrices).
%
%   A may name a NESTED attribute or a flat one. A flat attribute reads
%   the r!*C(K, r) / C(K, r) pair the build enumerates, with the r!
%   dropped on an ordered attribute, and its contraction work is the
%   product of the two counts -- which is how the cost model prices an
%   ordered flat companion of a nested attribute on the centres law.
%
%   This is a thin accessor over INTERNAL.NESTEDCONTRACT's opts.termsOnly
%   mode -- the counts come from that file's own tupleCounts, recipeWork
%   and quadNodes helpers, so the calibration harness
%   (TESTS/BENCH_NESTED_COST), the dispatch and the cost model
%   (INTERNAL.NESTEDCOST) cannot drift apart. It computes nothing and
%   takes no route.
%
%   Mirror of Python mpt/_tensor/_nested_cost.nested_attr_terms together
%   with predict_nested_pairwise_kernel_size.

    if nargin < 3 || isempty(a); a = 1; end
    if nargin < 4; truncationSigmas = []; end
    if nargin < 5 || isempty(skipXX); skipXX = false; end
    if nargin < 6 || isempty(skipYY); skipYY = false; end
    opts = struct('termsOnly', true, 'termsAttr', a, ...
                  'termsSkipXX', logical(skipXX), ...
                  'termsSkipYY', logical(skipYY));
    out = internal.nestedContract(densX, densY, 'cosine', ...
                                  truncationSigmas, false, opts);
    terms  = out.terms;
    bulger = out.bulger;
    info   = out.info;
end
