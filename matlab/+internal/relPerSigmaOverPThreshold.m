function sop = relPerSigmaOverPThreshold(truncationSigmas)
%RELPERSIGMAOVERPTHRESHOLD  sigma/P above which the wrapped-difference
%   relative-periodic form is inadmissible.
%
%   SOP = INTERNAL.RELPERSIGMAOVERPTHRESHOLD() uses the toolbox default
%   accuracy; INTERNAL.RELPERSIGMAOVERPTHRESHOLD(TS) uses the width TS.
%
%   Two tests, the stricter winning.
%
%   The accuracy test is the toolbox's usual one: the departure of the
%   approximating form from the transposition average that defines the
%   measure must sit inside the floor truncationSigmas implies, judged
%   as an absolute error on the value scale. The relative periodic
%   density is defined as the transposition average of the absolute
%   periodic density; the wrapped-difference kernel approximates it,
%   coinciding only as sigma/P -> 0, and is used because it is cheaper.
%
%   The positive-definiteness test is a fixed ceiling. Beyond it the
%   approximating kernel stops being positive-definite, its induced
%   cosine similarity exceeds 1, and the quantity is not a similarity at
%   all --- a failure of admissibility rather than of accuracy, which no
%   accuracy setting has authority to loosen.
%
%   Accuracy is the binding test in practice: it gives 0.03 at the
%   factory default and at every tighter setting, 0.02 at
%   truncationSigmas = 8, and 0.055 at the loosest, where the ceiling
%   clamps it to 0.05. A single constant could only be right at one
%   setting; the shipped 0.03 was the value at the default.
%
%   The table is the calibration and no functional form is fitted to it.
%   The largest entry inside the floor is taken rather than
%   interpolated, so the answer is always one the measurements support.
%   Measured by tools/calibrate_sigma_over_p.py over tuple orders 2 to
%   4, value counts 4 to 12, and six weight profiles. The two entries
%   below 0.04 sit at floating-point noise rather than at a measured
%   departure. The departure peaks at r = 3 rather than at the largest
%   order, so a calibration taken at r = 2 alone would be too loose.
%
%   Constants are shared with the Python twin
%   _orbit_sigma_over_p_threshold: the departure is a property of the
%   measures, not of either implementation.
%
%   See also INTERNAL.TRUNCATIONFLOOR, INTERNAL.SELECTMAEVAL.

    if nargin < 1
        truncationSigmas = [];
    end

    % sigma/P against the worst measured departure at that sigma/P.
    % The table does not set the limit at every accuracy setting: at
    % truncationSigmas = 4 the departures admit 0.055, and at 2 they
    % admit 0.100, both cut to 0.05 by PD_CEILING below. The table
    % binds at truncationSigmas = 5 and tighter, the ceiling at 4 and
    % looser.
    DEPARTURE = [0.020, 1.67e-16
                 0.030, 4.14e-14
                 0.040, 3.27e-08
                 0.050, 1.82e-05
                 0.055, 1.09e-04
                 0.060, 3.99e-04
                 0.065, 1.03e-03
                 0.070, 3.08e-03
                 0.080, 1.10e-02
                 0.100, 4.07e-02];

    % Hard ceiling from positive-definiteness. Three searches over the
    % same grid have first reached a cosine above 1 at sigma/P = 0.06,
    % 0.07 and 0.08 respectively. A search only ever bounds the onset
    % from above --- failing to find a violation proves nothing, and the
    % spread across runs shows how little a single onset settles --- so
    % the ceiling sits below the earliest onset anyone has found. It is
    % not merely a backstop: at truncationSigmas = 4 and looser it is
    % the binding test.
    PD_CEILING = 0.05;

    floorVal = internal.truncationFloor(truncationSigmas);
    admissible = DEPARTURE(DEPARTURE(:, 2) <= floorVal, 1);
    if isempty(admissible)
        sop = DEPARTURE(1, 1);
    else
        sop = max(admissible);
    end
    sop = min(sop, PD_CEILING);
end
