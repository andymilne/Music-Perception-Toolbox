function maybeWarnAbsPerSingleImage(sigma, isRel, isPer, period)
%MAYBEWARNABSPERSINGLEIMAGE  Warn once per offending absolute-periodic
%   attribute at density construction.
%
%   Defensive throughout: sigma may carry an anisotropic kernel
%   covariance rather than a scalar, and the mode vectors may be ragged
%   or absent on partially built structs, so anything that does not
%   resolve to a finite positive scalar pair is skipped rather than
%   raised on. A density is metadata; constructing one must not fail
%   because a diagnostic could not be evaluated.
%
%   Mirror of Python density._warn_if_abs_per_single_image.
    THRESHOLD = 0.05;   % _ABS_PER_SIGMA_OVER_P_THRESHOLD
    try
        if isempty(isPer) || isempty(period) || isempty(sigma)
            return;
        end
        perV = logical(isPer(:));
        if isempty(isRel)
            relV = false(size(perV));
        else
            relV = logical(isRel(:));
        end
        sigV = sigma;
        if iscell(sigV) || isstruct(sigV)
            return;   % anisotropic kernel covariance; not a scalar sigma
        end
        sigV = sigV(:);
        perdV = period(:);
        for a = 1:numel(perV)
            if ~perV(a)
                continue;
            end
            if a <= numel(relV) && relV(a)
                continue;
            end
            if a > numel(sigV) || a > numel(perdV)
                continue;
            end
            s = double(sigV(a));
            P = double(perdV(a));
            if ~isfinite(s) || ~isfinite(P) || P <= 0
                continue;
            end
            if s / P > THRESHOLD
                internal.warnAbsPerSingleImage(s / P);
            end
        end
    catch
        % Never let a diagnostic break construction.
        return;
    end
end
