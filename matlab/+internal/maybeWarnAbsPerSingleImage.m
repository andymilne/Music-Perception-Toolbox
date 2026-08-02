function maybeWarnAbsPerSingleImage(sigma, isRel, isPer, period, wrap)
%MAYBEWARNABSPERSINGLEIMAGE  Warn once per offending absolute-periodic
%   attribute at density construction, but only when the user has opted
%   that attribute into ``wrap = 'single-image'``.
%
%   The default full-image measure is positive definite by construction,
%   so no warning is needed in the ordinary case.
%
%   Defensive throughout: sigma may carry an anisotropic kernel
%   covariance rather than a scalar, and the mode vectors may be ragged
%   or absent on partially built structs, so anything that does not
%   resolve to a finite positive scalar pair is skipped rather than
%   raised on. A density is metadata; constructing one must not fail
%   because a diagnostic could not be evaluated.
%
%   Mirror of Python density._warn_if_abs_per_single_image.
    if nargin < 5
        wrap = [];
    end
    THRESHOLD = internal.absPerSigmaOverPThreshold();
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
            % Fire only when the user has opted this attribute into
            % single-image. Legacy callers without a wrap vector stay
            % silent (implicit default is full-image).
            if isempty(wrap)
                continue;
            end
            if iscell(wrap)
                if a > numel(wrap)
                    continue;
                end
                wrapA = char(wrap{a});
            elseif ischar(wrap) || (isstring(wrap) && isscalar(wrap))
                wrapA = char(wrap);
            else
                wrapArr = cellstr(wrap);
                if a > numel(wrapArr)
                    continue;
                end
                wrapA = wrapArr{a};
            end
            if ~strcmp(wrapA, 'single-image')
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
