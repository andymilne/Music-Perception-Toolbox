function tf = weightIsLive(w)
%WEIGHTISLIVE  Mask of weights that contribute to a density.
%
%   TF = internal.weightIsLive(W) returns a logical mask, the same size as
%   W, that is true where a weight contributes. A weight contributes iff it
%   is finite and of nonzero magnitude: NaN (a structurally absent value) and
%   0 (present but zero-weighted, e.g. hard-zeroed outside a window's
%   truncation support) both fail the test.
%
%   This is the single definition of a "live" weight, shared by the
%   event-level prune (internal.prunedExpTens) and the pre-build seam prune
%   (internal.pruneDeadEvents). It is the MATLAB counterpart of the Python
%   density helper `_weight_is_live`.

    tf = isfinite(w) & abs(w) > 0;
end
