function [tf, logRatio] = orbitCostModel(r, K, B)
%ORBITCOSTMODEL  Is the Mobius route the faster combine at this shape?
%
%   TF = internal.orbitCostModel(R, K, B) returns true when the Mobius
%   (orbit) route is predicted faster than enumeration for a combine at
%   tuple size R, member count K, and batch extent B.
%
%   [TF, LOGRATIO] = ... also returns the predicted log of
%   t_orbit / t_enum. Negative favours the Mobius route. Returning the
%   margin as well as the verdict lets a caller report how close the
%   decision was, and lets a harness compare predicted against measured
%   on a continuous scale rather than as a win/lose label.
%
%   Form
%   ----
%   Each route's time was fitted as a power law in the quantities it
%   works on, and the two fits subtracted, so only their difference
%   appears here:
%
%       log(t_orbit / t_enum) = c0 + c1 log|Omega_r| + c2 log K
%                                  + c3 log B + c4 log(Tx Ty) + c5 log r
%
%   with Tx = K!/(K-r)! and Ty = nchoosek(K, r) the tuple counts
%   enumeration materialises, and |Omega_r| the number of orbit classes
%   the Mobius route loops over.
%
%   Why not a flop count
%   --------------------
%   The criterion this replaces compared flop counts, |Omega_r| K K
%   against C(K,r)^2 r!. Flop counts do not predict time here: the two
%   routes have very different cost per operation, one being a loop over
%   orbit classes and the other a single materialise-and-reduce. Measured
%   against 444 timed cells the flop criterion placed the crossover
%   within one of its true K in 6 of 21 (r, B) combinations, missing by
%   up to 15; this model manages 19 of 21.
%
%   Why the batch extent is an argument
%   -----------------------------------
%   The crossover moves by up to 11 in K across the batch range --- for
%   r = 2, from K = 19 at B = 1 down to K = 8 at B = 2048 --- because the
%   two routes amortise differently over the batch. A predicate without
%   B cannot express that.
%
%   Calibration
%   -----------
%   Only c0 is machine-specific; c1 to c5 are scaling exponents. The
%   shipped value was measured on one machine, so it will be somewhat
%   wrong elsewhere, in the same way the constant it replaces was. It is
%   isolated as a single number so that recalibrating means measuring one
%   quantity rather than repopulating a table: it is the default
%   orbitCostIntercept, and tools/calibrateOrbitIntercept measures it.
%
%   Accuracy: over 484 timed cells the predicted time ratio sits within a
%   factor of about 2.5 of the measured one typically, with occasional
%   cells further out. Adequate for the purpose, since the two routes
%   differ by 10x to 200x away from the crossover and near it either
%   choice costs little. Refitting the difference directly, and adding a
%   second run's cells, left the coefficients and the crossover placement
%   unchanged.
%
%   Known weak spot: at r = 2 and low batch the model is conservative by
%   up to 5 in K, preferring enumeration past the point where the Mobius
%   route overtakes. That costs speed, not accuracy. At r = 8 and high
%   batch it predicts a crossover one higher than the trend in its own
%   data implies, and is the least trustworthy cell.
%
%   See also internal.nestedContract, tools/calibrateOrbitCrossover.
    arguments
        r (1,1) {mustBeInteger, mustBePositive}
        K (1,1) {mustBeInteger, mustBePositive}
        B (1,1) {mustBeInteger, mustBePositive} = 1
    end

    if r > K
        tf = false; logRatio = Inf;   % no r-tuples to form
        return;
    end
    if r < 2
        tf = false; logRatio = Inf;   % an r = 1 factor is exact either way
        return;
    end

    % Small tuple sizes get one element of headroom. Every hold-out miss
    % sat at r <= 3, where the fit has the fewest orbit classes to work
    % with and switched to the Mobius route one or two K early. Testing
    % one K lower delays the switch by one.
    Keff = K;
    if r <= 3
        Keff = max(r, K - 1);
    end

    logRatio = costLogRatio(r, Keff, B);
    tf = logRatio < 0;
end


function v = costLogRatio(r, K, B)
    % Fitted 2026-07-30 against 444 timed cells, r = 2..8, K = r..22,
    % B = 1..2048, absolute and relative-periodic modes (which timed
    % identically). Leave-one-tuple-size-out validation reproduced the
    % measured crossover within one in K in 19 of 21 combinations, and
    % held-out performance matched in-sample, indicating the fit captured
    % the scaling rather than memorising individual tuple sizes.
    C = [ mptDefaults('orbitCostIntercept'), ...  % machine-specific term
          1.0708, ...   % log |Omega_r|
          1.4033, ...   % log K
         -0.3608, ...   % log B
         -0.8188, ...   % log (Tx Ty)
         -0.2261];      % log r
    logTxTy = gammaln(K + 1) - gammaln(K - r + 1) ...    % log Tx
            + gammaln(K + 1) - gammaln(r + 1) - gammaln(K - r + 1);  % log Ty
    v = C(1) ...
      + C(2) * log(numel(mobius.getOrbitTable(r))) ...
      + C(3) * log(K) ...
      + C(4) * log(B) ...
      + C(5) * logTxTy ...
      + C(6) * log(r);
end
